import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console

load_dotenv()

console = Console()
username = os.getlogin()

def generate():
    client = genai.Client(
        api_key=os.environ.get("API_KEY"),
    )

    model = "gemini-2.5-flash-preview-04-17"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""{i}"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        console.print(f"[dark_orange]gemini@ai:[/dark_orange] {chunk.text}", end="\n")

if __name__ == "__main__":
    while True:
        i = console.input(f"[green]you@{username}[/green][blue]_>[/blue] ")
        generate()
