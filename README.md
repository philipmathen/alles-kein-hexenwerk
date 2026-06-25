# How to build a coding agent

Eine Schritt-für-Schritt-Demo, die zeigt, wie ein Coding-Agent **unter der Haube**
funktioniert. Jeder Ordner ist eine eigenständige, lauffähige Stufe. Von einer
Stufe zur nächsten kommt genau eine neue Fähigkeit dazu – so sieht man, was jedes
einzelne Teil beiträgt.

| Stufe | Ordner | Was neu ist |
|-------|--------|-------------|
| 1 | [`01-chat`](01-chat) | Reiner Chat-Loop mit dem LLM – noch keine Tools |
| 2 | [`02-read-file`](02-read-file) | Tool-Infrastruktur + `read_file` |
| 3 | [`03-list-files`](03-list-files) | `list_files` – der Agent kann das Dateisystem erkunden |
| 4 | [`04-edit-file`](04-edit-file) | `edit_file` – der Agent kann Dateien schreiben/ändern |

## Setup

Alle Stufen teilen sich ein Go-Modul und eine `.env` Datei im Repo-Root.

Eine `.env` mit deinem Anthropic API Key anlegen:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Jede Stufe sucht ab dem Arbeitsverzeichnis aufwärts nach dieser `.env` – sie wird
also gefunden, egal aus welchem Ordner das Programm gestartet wird.

## Eine Stufe starten

Aus dem Repo-Root:

```bash
go run ./01-chat
go run ./02-read-file
go run ./03-list-files
go run ./04-edit-file
```

Oder direkt aus dem Ordner einer Stufe:

```bash
cd 03-list-files
go run .
```

Nachricht eintippen und Enter drücken. `Strg+C` zum Beenden.

## Die Idee in einem Satz

Ein Agent ist nur eine **Schleife**: die Konversation an das Modell schicken, die
angeforderten Tools ausführen, das Ergebnis zurückfüttern, wiederholen – alles
andere sind Tools.
