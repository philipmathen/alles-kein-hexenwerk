// Schritt 2: Tools einführen + erstes Tool "read_file".
//
// Neu gegenüber Schritt 1:
//   - ToolDefinition: ein Tool besteht aus Name, Beschreibung, Input-Schema
//     und der auszuführenden Funktion.
//   - GenerateSchema: erzeugt per Reflection aus einem Go-Struct das
//     JSON-Schema, das dem Modell als "Vertrag" für die Tool-Argumente dient.
//   - Die Agent-Schleife verarbeitet jetzt tool_use-Blöcke: das Tool wird
//     ausgeführt und das Ergebnis direkt wieder ins Modell zurückgegeben.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

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

	tools := []ToolDefinition{ReadFileDefinition}
	agent := NewAgent(&client, getUserMessage, tools)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

// loadEnv sucht ab dem Arbeitsverzeichnis aufwärts nach einer .env Datei.
// Dadurch funktioniert jeder Schritt – egal ob aus dem Repo-Root
// (go run ./02-read-file) oder direkt aus dem Ordner (go run .) gestartet.
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

	fmt.Println("=== Stufe 2: Tool-Infrastruktur + read_file ===")
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
	// JSON-Rohdaten kommen rein
	readFileInput := ReadFileInput{}
	// JSON wird in das Struct geladen
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		return "", err
	}

	// Wenn Struct gültig geladen, dann File am Path lesen
	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}
	// Inhalt des Files als string zurückgeben
	return string(content), nil
}
