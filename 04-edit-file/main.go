// Schritt 4: Tool "edit_file" hinzufügen.
//
// Neu gegenüber Schritt 3: Der Agent kann jetzt nicht nur lesen, sondern auch
// schreiben. edit_file ersetzt 'old_str' durch 'new_str' in einer Datei – und
// legt die Datei neu an, wenn 'old_str' leer ist. Damit wird aus dem reinen
// "Leser" ein echter Coding-Agent, der Änderungen am Code vornehmen kann.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/invopop/jsonschema"
	"github.com/lpernett/godotenv"
)

func main() {
	loadEnv()

	client := anthropic.NewClient()

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []ToolDefinition{ReadFileDefinition, ListFilesDefinition, EditFileDefinition}
	agent := NewAgent(&client, getUserMessage, tools)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

// loadEnv sucht ab dem Arbeitsverzeichnis aufwärts nach einer .env Datei.
// Dadurch funktioniert jeder Schritt – egal ob aus dem Repo-Root
// (go run ./04-edit-file) oder direkt aus dem Ordner (go run .) gestartet.
func loadEnv() {
	dir, err := os.Getwd()
	if err != nil {
		log.Fatalf("could not determine working directory: %v", err)
	}
	for {
		envPath := filepath.Join(dir, ".env")
		if _, err := os.Stat(envPath); err == nil {
			if err := godotenv.Load(envPath); err != nil {
				log.Fatalf("Error loading .env file: %v", err)
			}
			return
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	log.Fatal(".env file not found")
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
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

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with claude (ctrl + c to quit)")

	// Flag ob der User dran ist
	readUserInput := true
	// Agent Loop
	for {
		// Input vom User entgegennehmen
		if readUserInput {
			fmt.Print("\x1b[38;5;39mUser\x1b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}
			userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
			conversation = append(conversation, userMessage)
		}

		// Antwort vom Modell generieren und an die Konversation anhängen
		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		conversation = append(conversation, message.ToParam())

		toolResults := []anthropic.ContentBlockParamUnion{}

		// Antwort vom Modell prüfen
		for _, content := range message.Content {
			switch content.Type {
			// Wenn es Text ist, im Terminal anzeigen
			case "text":
				fmt.Printf("\x1b[38;5;208mClaude\x1b[0m: %s\n", content.Text)
			// Wenn es ein Toolcall ist, das Tool mit dem Input ausführen und das Ergebnis speichern
			case "tool_use":
				result := a.executeTool(content.ID, content.Name, content.Input)
				toolResults = append(toolResults, result)
			}
		}
		// Wenn es keine Ergebnisse aus den Tools gibt, ist der User wieder dran
		if len(toolResults) == 0 {
			readUserInput = true
			continue
		}
		// Wenn ein Tool Ergebnisse geliefert hat, wird das Ergebnis in den
		// Kontext eingefügt und direkt wieder in das LLM gefüttert
		readUserInput = false
		conversation = append(conversation, anthropic.NewUserMessage(toolResults...))
	}
	return nil
}

func (a *Agent) executeTool(id, name string, input json.RawMessage) anthropic.ContentBlockParamUnion {
	var toolDef ToolDefinition
	var found bool
	// Verfügbare Tools des Agenten prüfen
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

	// Tool und Input ausgeben
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

// Tools

// ToolDefinition beschreibt ein Tool, das der Agent benutzen kann.
type ToolDefinition struct {
	Name        string
	Description string
	InputSchema anthropic.ToolInputSchemaParam
	Function    func(input json.RawMessage) (string, error)
}

// GenerateSchema baut aus einem Go-Type per Reflection das JSON-Schema.
// Das Schema ist der "Vertrag" mit dem LLM: es beschreibt, wie die
// Tool-Argumente aussehen müssen. Die jsonschema-Struct-Tags am Input-Struct
// liefern dabei die Beschreibungen der einzelnen Felder.
func GenerateSchema[T any]() anthropic.ToolInputSchemaParam {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)

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

func ReadFile(input json.RawMessage) (string, error) {
	readFileInput := ReadFileInput{}
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		return "", err
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}
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
		return "", err
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
		return "", err
	}
	return "OK", nil
}

func createNewFile(filePath, content string) (string, error) {
	dir := path.Dir(filePath)
	if dir != "." {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed  to create directory %w", err)
		}
	}
	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create File")
	}

	return fmt.Sprintf("Successfully created file %s", filePath), nil
}
