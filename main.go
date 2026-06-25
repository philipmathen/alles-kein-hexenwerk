package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/invopop/jsonschema"
	"github.com/lpernett/godotenv"
)

func init() {
	if _, err := os.Stat(".env"); err == nil {
		err := godotenv.Load()
		if err != nil {
			log.Fatalf("Error loading .env file: %v", err)
		}
	} else {
		log.Fatal(".env File not found")
	}
}

func main() {

	client := anthropic.NewClient()

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}
	tools := []ToolDefinition{ReadFileDefinition, ListFilesDefinition, EditFileDefinition, GitCommandDefinition}
	agent := NewAgent(&client, getUserMessage, tools)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}

}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with claude (ctrl + c to quit)")

	// Flag ob der User dran ist
	readUserInput := true
	// Agent Loop
	for {
		//Input vom User entgegennehmen
		if readUserInput {
			fmt.Print("\x1b[38;5;39mUser\x1b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}
			userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
			conversation = append(conversation, userMessage)
		}

		// Antwort vom Modell genrieren und an die Konversation anhängen
		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		conversation = append(conversation, message.ToParam())

		toolResults := []anthropic.ContentBlockParamUnion{}

		//Antwort vom Modell prüfen
		for _, content := range message.Content {
			switch content.Type {
			// Wenn es text ist im Terminal anzeigen
			case "text":
				fmt.Printf("\x1b[38;5;208mClaude\x1b[0m: %s\n", content.Text)
				// Wenn es ein Toolcall ist, das Tool mit dem Input ausführen und das Ergebnis speichern
			case "tool_use":
				result := a.executeTool(content.ID, content.Name, content.Input)
				toolResults = append(toolResults, result)
			}
		}
		// Wenn es keine Ergebnisse aus den Tools gibt ist der User wieder dran
		if len(toolResults) == 0 {
			readUserInput = true
			continue
		}
		// Wenn ein Tool Ergebnisse geliefert hat, dann wird das Egebnis
		// in den Kontext eingefügt und direkt wieder in das LLM gefüttert
		readUserInput = false
		conversation = append(conversation, anthropic.NewUserMessage(toolResults...))
	}
	return nil
}

func (a *Agent) executeTool(id, name string, input json.RawMessage) anthropic.ContentBlockParamUnion {
	var toolDef ToolDefinition
	var found bool
	//verfügabre Tools des Agenten prüfen
	for _, tool := range a.tools {
		if tool.Name == name {
			toolDef = tool
			found = true
			break
		}
	}

	if !found {
		return anthropic.NewToolResultBlock(id, "tool not found", true)
	}

	// Tool und input ausgeben
	fmt.Printf("\x1b[38;5;46mTool\x1b[0m: %s(%s)\n", name, input)

	// Funktion des Tools ausführen
	response, err := toolDef.Function(input)
	if err != nil {
		return anthropic.NewToolResultBlock(id, err.Error(), true)
	}
	// Ergebnis ausgeben und zurückliefern
	fmt.Printf("\x1b[38;5;82mResult\x1b[0m: %s\n", response)
	return anthropic.NewToolResultBlock(id, response, false)
}

// Tools
// ToolDefinition
type ToolDefinition struct {
	Name        string
	Description string
	InputSchema anthropic.ToolInputSchemaParam
	Function    func(input json.RawMessage) (string, error)
}

// Json Schema als Vertrag mit dem LLM wie ein tool aussehen muss
// jsonschema verarbeitet die jsonschema struct tags die oben im struct definert wurden
func GenerateSchema[T any]() anthropic.ToolInputSchemaParam {
	// Reflektor ist baut Schema aus Type (wird hier konfiguriert)
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	//Schema erstellen
	schema := reflector.Reflect(v)

	schemaBytes, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		panic(err)
	}
	schemaString := string(schemaBytes)
	fmt.Println(schemaString)

	// Properties in anthropic kompatibles Schema wrappen
	return anthropic.ToolInputSchemaParam{
		Properties: schema.Properties,
	}
}

// Tool: ReadFile
var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use this when you want to see whats in the file. Don´t use it on directory names.",
	InputSchema: ReadFileInputSchema,
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The realtive path of a file in the working directory"`
}

var ReadFileInputSchema = GenerateSchema[ReadFileInput]()

// Function des ReadFile tools
func ReadFile(input json.RawMessage) (string, error) {
	// Json rohdaten kommen rein
	readFileInput := ReadFileInput{}
	//Json wird in das struct geldaden
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		panic(err)
	}

	//Wenn struct gültig geladen, dann File am Path lesen
	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}
	// Inhalt des Files als string zurückgeben
	return string(content), nil
}

// Tool: ListFiles
var ListFilesDefinition = ToolDefinition{
	Name:        "list_files",
	Description: "List files and directory at a given path. If no path is provided, lists files in the current directory",
	InputSchema: ListFilesInputSchema,
	Function:    ListFiles,
}

type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to current directory if not provided."`
}

var ListFilesInputSchema = GenerateSchema[ListFilesInput]()

func ListFiles(input json.RawMessage) (string, error) {
	listFilesInput := ListFilesInput{}
	err := json.Unmarshal(input, &listFilesInput)
	if err != nil {
		panic(err)
	}
	dir := "."
	if listFilesInput.Path != "" {
		dir = listFilesInput.Path
	}

	var files []string
	err = filepath.WalkDir(dir, func(path string, dirEntry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		relativePath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		if relativePath != "." {
			// versteckte Dateien ausblenden
			fileInfo, _ := dirEntry.Info()
			if fileInfo.Name()[0] == '.' {
				return nil
			}
			if dirEntry.IsDir() {
				files = append(files, relativePath+"/")
			} else {
				files = append(files, relativePath)
			}
		}
		return nil
	})

	if err != nil {
		return "", err
	}

	result, err := json.Marshal(files)
	if err != nil {
		return "", err
	}
	return string(result), nil
}

// Tool: EditFile
var EditFileDefinition = ToolDefinition{
	Name: "edit_file",
	Description: `Make edits to a text file.

Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.
`,
	InputSchema: EditFileInputSchema,
	Function:    EditFile,
}

type EditFileInput struct {
	Path   string `json:"path" jsonschema_description:"The path to the file"`
	OldStr string `json:"old_str" jsonschema_description:"Text to search for - must match exactly and must only have one match exactly"`
	NewStr string `json:"new_str" jsonschema_description:"Text to replace old_str with"`
}

var EditFileInputSchema = GenerateSchema[EditFileInput]()

func EditFile(input json.RawMessage) (string, error) {
	editFileInput := EditFileInput{}
	err := json.Unmarshal(input, &editFileInput)
	if err != nil {
		return "", err
	}

	if editFileInput.Path == "" || editFileInput.OldStr == editFileInput.NewStr {
		return "", fmt.Errorf("invalid input")
	}
	content, err := os.ReadFile(editFileInput.Path)
	if err != nil {
		if os.IsNotExist(err) && editFileInput.OldStr == "" {
			return createNewFile(editFileInput.Path, editFileInput.NewStr)
		}
		return "", err
	}
	oldContent := string(content)
	newContent := strings.Replace(oldContent, editFileInput.OldStr, editFileInput.NewStr, -1)

	if oldContent == newContent && editFileInput.OldStr != "" {
		return "", fmt.Errorf("old_str not found in file")
	}

	err = os.WriteFile(editFileInput.Path, []byte(newContent), 0644)
	if err != nil {
		return "", nil
	}
	return "OK", nil
}

func createNewFile(filePpath, content string) (string, error) {
	dir := path.Dir(filePpath)
	if dir != "." {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed  to create directory %w", err)
		}
	}
	err := os.WriteFile(filePpath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create File")
	}

	return fmt.Sprintf("Successfully created file %s", filePpath), nil
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (anthropic.Message, error) {
	anthropicTools := []anthropic.ToolUnionParam{}

	for _, tool := range a.tools {
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Description),
				InputSchema: tool.InputSchema,
			},
		})
	}

	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeHaiku4_5,
		MaxTokens: int64(1024),
		Messages:  conversation,
		Tools:     anthropicTools,
	})
	return *message, err
}

func NewAgent(
	client *anthropic.Client,
	getUserMessage func() (string, bool),
	tools []ToolDefinition,
) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
	}
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
}

// Tool: GitCommand (created by the agent itself)
var GitCommandDefinition = ToolDefinition{
	Name:        "git_command",
	Description: "Execute a git command. Supports: status, add, commit, push, pull, log, diff. Use this to manage version control operations.",
	InputSchema: GitCommandInputSchema,
	Function:    GitCommand,
}

type GitCommandInput struct {
	Command string `json:"command" jsonschema_description:"Git command to execute: status, add, commit, push, pull, log, diff"`
	Args    string `json:"args,omitempty" jsonschema_description:"Arguments for the git command (e.g., file paths, commit messages)"`
}

var GitCommandInputSchema = GenerateSchema[GitCommandInput]()

func GitCommand(input json.RawMessage) (string, error) {
	gitInput := GitCommandInput{}
	err := json.Unmarshal(input, &gitInput)
	if err != nil {
		return "", err
	}

	// Allowed git commands for safety
	allowedCommands := map[string]bool{
		"status": true,
		"add":    true,
		"commit": true,
		"push":   true,
		"pull":   true,
		"log":    true,
		"diff":   true,
	}

	if !allowedCommands[gitInput.Command] {
		return "", fmt.Errorf("command '%s' not allowed. Use one of: status, add, commit, push, pull, log, diff", gitInput.Command)
	}

	var cmd *exec.Cmd
	if gitInput.Args != "" {
		cmd = exec.Command("git", gitInput.Command, gitInput.Args)
	} else {
		cmd = exec.Command("git", gitInput.Command)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()
	if err != nil {
		if stderr.Len() > 0 {
			return stderr.String(), err
		}
		return "", err
	}

	return stdout.String(), nil
}
