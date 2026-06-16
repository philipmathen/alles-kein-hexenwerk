package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

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
	tools := []ToolDefinition{ReadFileDefinition}
	agent := NewAgent(&client, getUserMessage, tools)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}

}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with claude (ctrl + c to quit)")

	for {
		fmt.Print("\x1b[94mUser\x1b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}
		userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation := append(conversation, userMessage)

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		for _, content := range message.Content {
			switch content.Type {
			case "text":
				fmt.Printf("\x1b[93mClaude\x1b[0m: %s\n", content.Text)
			}
		}
	}
	return nil
}

type ToolDefinition struct {
	Name        string
	Description string
	InputSchema anthropic.ToolInputSchemaParam
	Function    func(input json.RawMessage) (string, error)
}

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

// Json Schema als Vertrag mit dem LLM wie ein toolcall aussehen muss
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
	// Properties in anthropic kompatibles Schema wrappen
	return anthropic.ToolInputSchemaParam{
		Properties: schema.Properties,
	}
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
