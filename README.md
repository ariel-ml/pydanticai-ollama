# Pydantic AI Ollama Wrapper

This project provides a custom `OllamaModel` wrapper for the Pydantic AI framework, allowing seamless integration and fine-grained control over Ollama models. It addresses the limitations of using the generic `OpenAIModel` for Ollama-specific parameters and features, including faster response times and fewer API calls.

## Features

- Dedicated `OllamaModel` class for Pydantic AI.
- Comprehensive `OllamaModelSettings` for all Ollama-specific parameters, including `temperature`, `num_predict`, `top_k`, `top_p`, `stop`, and advanced `think` (reasoning) mode.
- Integration with the standard Ollama Python client.
- Easy to use with Pydantic AI's `Agent`.
- **Full support for streaming responses.**
- **Function Calling / Tool Use**: Define and use tools for structured interactions with the model.
- **Structured Output**: Generate responses directly as Pydantic models (JSON format).
- **Multi-modal Input**: Send images along with text prompts to supported Ollama models.

## Quick Start

**Install using pip**

```bash
pip install pydanticai-ollama
```

**Install using uv**
```bash
uv add pydanticai-ollama
```

## Installation

1.  **Clone the repository (if you haven't already):**

    ```bash
    git clone https://github.com/ariel-ml/pydanticai-ollama.git
    cd pydanticai-ollama
    ```

2.  **Create and activate a virtual environment using `uv`:**

    ```bash
    uv venv
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate
    ```

3.  **Install the project dependencies:**

    ```bash
    uv sync
    ```

    This command installs the project in editable mode, making the `OllamaModel` and related components available to your Python environment.

## Usage

First, ensure you have an Ollama server running and a model downloaded (e.g., `ollama run qwen3:4b-instruct` or `ollama run gemma3:latest` for multi-modal capabilities).

### Basic Text Generation

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydanticai_ollama.models.ollama import OllamaModel
from pydanticai_ollama.providers.ollama import OllamaProvider
from pydanticai_ollama.settings.ollama import OllamaModelSettings

# 1. Define your output type (optional, but recommended for structured responses)
class CityLocation(BaseModel):
    city: str
    country: str

# 2. Configure OllamaModelSettings with desired parameters
#    You can set any parameter defined in OllamaModelSettings (e.g., temperature, num_predict, top_k, etc.)
ollama_settings = OllamaModelSettings(
    temperature=0.7,
    num_predict=128,
    top_k=40,
    top_p=0.9
)

# 3. Initialize OllamaProvider with your Ollama server's base URL
#    Default is "http://localhost:11434"
ollama_provider = OllamaProvider(base_url="http://localhost:11434")

# 4. Create an OllamaModel instance
#    Replace 'llama2' with the name of the Ollama model you want to use
ollama_model = OllamaModel(
    model_name='qwen3:4b-instruct',
    provider=ollama_provider,
    settings=ollama_settings,
)

# 5. Create a Pydantic AI Agent with your OllamaModel
agent = Agent(ollama_model, output_type=CityLocation)

# 6. Run the agent
async def main():
    result = await agent.run('Where were the olympics held in 2012?')
    print(result.output)
    print(result.usage())

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming Responses

The `OllamaModel` fully supports streaming responses. When you run an agent with `stream=True` (or if the underlying model supports streaming by default), you can iterate over the response:

```python
import asyncio
from pydantic_ai import Agent
from pydanticai_ollama.models.ollama import OllamaModel
from pydanticai_ollama.providers.ollama import OllamaProvider

async def streaming_example():
    ollama_model = OllamaModel(
        model_name='qwen3:4b-instruct',
        provider=OllamaProvider(base_url="http://localhost:11434")
    )
    agent = Agent(ollama_model)

    print("Streaming response:")
    async for chunk in agent.run_stream('Tell me a long story about a space cat.'):
        if chunk.output:
            print(chunk.output, end='', flush=True)
    print("\nStreaming finished.")

if __name__ == "__main__":
    asyncio.run(streaming_example())
```

### Function Calling / Tool Use

Define tools using Pydantic models and let the Ollama model call them:

```python
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydanticai_ollama.models.ollama import OllamaModel
from pydanticai_ollama.providers.ollama import OllamaProvider

# 1. Define a Pydantic model for your tool's arguments
class GetCurrentWeatherArgs(BaseModel):
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(default="celsius", description="The unit of temperature (celsius or fahrenheit) to return")

async def tool_use_example():
    ollama_model = OllamaModel(model_name='qwen3:4b-instruct') # Ensure model supports function calling (e.g., qwen)
    agent = Agent(ollama_model)

    @agent.tool
    def get_current_weather(ctx: RunContext[str], args: GetCurrentWeatherArgs) -> str:
        """Get the current weather in a given location."""
        # In a real application, this would call an external weather API
        print(f"Tool args: {args}")
        if "San Francisco" in args.location:
            return f"24 degrees {args.unit} and sunny in San Francisco."
        else:
            return f"28 degrees {args.unit} and cloudy in {args.location}."

    print("Tool use example:")
    result = await agent.run("What's the weather like in San Francisco?")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(tool_use_example())
```

### Structured Output (JSON Mode)

Force the model to respond with a Pydantic model (JSON):

```python
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydanticai_ollama.models.ollama import OllamaModel
from pydanticai_ollama.providers.ollama import OllamaProvider

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")

async def structured_output_example():
    ollama_model = OllamaModel(model_name='llama2') # Ensure model supports JSON output
    agent = Agent(ollama_model, output_type=Joke)

    print("Structured output example:")
    joke_obj = await agent.run("Tell me a joke about a computer.")
    print(f"Setup: {joke_obj.output.setup}")
    print(f"Punchline: {joke_obj.output.punchline}")

if __name__ == "__main__":
    asyncio.run(structured_output_example())
```

### Multi-modal Input (Images)

Send an image along with your prompt (requires a multi-modal Ollama model like `llava`):

```python
import asyncio
from pydantic_ai import Agent, ImageUrl
from pydanticai_ollama.models.ollama import OllamaModel
from pydanticai_ollama.providers.ollama import OllamaProvider

async def multimodal_example():
    ollama_model = OllamaModel(model_name='llava') # Use a multi-modal model like LLaVA
    agent = Agent(ollama_model)

    # You can use a local file path or a URL
    image_path = "tests/assets/kiwi.png" # Ensure this path is correct or use a URL
    image_url = ImageUrl(url=image_path)

    print("Multi-modal example (describing an image):")
    result = await agent.run([image_url, "What is in this image?"])
    print(result.output)

if __name__ == "__main__":
    asyncio.run(multimodal_example())
```

## Development

### Running Tests

To run the unit tests, ensure your virtual environment is activated and then execute:

```bash
.venv/bin/python -m pytest tests/
```

**Note:** If you encounter `ModuleNotFoundError` during testing, ensure that the project is installed in editable mode (`uv sync`) and that `__init__.py` files are present in all package directories within `src/pydanticai_ollama`.

## Project Structure

```
. (project root)
├── src/
│   └── pydanticai_ollama/
│       ├── models/
│       │   └── ollama.py
│       ├── providers/
│       │   └── ollama.py
│       └── settings/
│           └── ollama.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── assets/
│   │   └── kiwi.png
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mock_async_stream.py
│   │   └── test_ollama.py
│   └── providers/
│       ├── __init__.py
│       └── test_ollama.py
├── pyproject.toml
├── README.md
├── uv.lock
└── .python-version
```
