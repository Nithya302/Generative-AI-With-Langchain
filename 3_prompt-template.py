from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatOpenAI(model="gpt-5-nano", temperature=0.0)

# Prompt template is a way to create dynamic prompts by filling in variables. 
# And it is useful when you want to create a prompt that can be reused with different inputs.

# Define the system prompt template
sys_prompt_template = PromptTemplate.from_template("""Answer the following question using the given context.
    If the context does not provide enough information, say 'I don't know'.
    Context: {context}
    Question: {question}"""
)

# generate the system prompt
sys_prompt = sys_prompt_template.invoke({
    "context": "The capital of India is New Delhi.",
    "question": "What is the capital of France?"
})

# Create the system message
response = model.invoke(sys_prompt)
print(response.content)


# ------------------------------------------------------------------
# Line-by-line explanation / comments for every import, class and call
# ------------------------------------------------------------------
#
# from dotenv import load_dotenv
#   - `dotenv` is a library that loads environment variables from a `.env` file into the process's environment.
#   - `load_dotenv()` reads a .env file in the current working directory (or a given path) and sets those key-value pairs
#     in `os.environ`. This is commonly used to load API keys or configuration without hard-coding them.
#   - Typical use here: you put your OpenAI (or other provider) API key in `.env`, e.g. `OPENAI_API_KEY=sk-...`,
#     then calling `load_dotenv()` ensures `os.environ["OPENAI_API_KEY"]` is available to libraries that read it.
#
# from langchain_openai.chat_models import ChatOpenAI
#   - Imports the ChatOpenAI class from LangChain's OpenAI integration module (package path may vary by LangChain
#     version; this import suggests a specific integration module).
#   - `ChatOpenAI` is a wrapper class that provides a conversational/chat interface to an OpenAI-like model.
#   - It exposes a simple API for creating a model instance with parameters (model name, temperature, etc.)
#   - Internally this class wraps request/response logic, token handling, retry logic, and the provider-specific API calls.
#
# from langchain_core.prompts import PromptTemplate
#   - Imports `PromptTemplate`, a helper class to define prompt *templates* with placeholders for variables.
#   - `PromptTemplate` lets you define a prompt with named variables (like `{context}` and `{question}`),
#     then render the final prompt by providing values for those variables.
#   - Typical capabilities:
#       * Template creation from a string
#       * Validation of required input variables
#       * Rendering (filling) into final text
#     Note: exact API names and module paths can change between LangChain versions.
#
# load_dotenv()
#   - Calls the function to load environment variables from a `.env` file.
#   - Ensures that any code or libraries that expect credentials or config in environment variables can read them.
#   - If you don't have a `.env` or keys in the real environment, the model may fail when trying to call the provider.
#
# model = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
#   - Creates an instance of the ChatOpenAI wrapper.
#   - Arguments:
#       * model="gpt-5-nano"
#           - The model identifier to use — this tells the wrapper which backend model to call.
#           - In practice you use a model name that your provider recognizes. If the model name is invalid or you have
#             no access, calls will fail.
#       * temperature=0.0
#           - Temperature controls randomness in generation. Range is typically [0.0, 1.0+] where:
#               - 0.0 → deterministic, low randomness (prefered for factual/precise outputs)
#               - higher values → more diverse / creative outputs
#           - Setting 0.0 means the model will try to be as deterministic as possible.
#   - The `model` object will have methods to invoke the model (e.g., `invoke`, depending on the wrapper's API).
#
# # Prompt template is a way to create dynamic prompts...
#   - This comment explains the general reason for using prompt templates: to reuse prompt skeletons with different
#     inputs instead of hard-coding multiple prompt strings.
#
# sys_prompt_template = PromptTemplate.from_template("""...""")
#   - Creates a PromptTemplate instance from a raw string template.
#   - The template contains placeholders in curly braces: `{context}` and `{question}`.
#   - The template text (here) instructs the model to:
#       1. Answer using the given context.
#       2. If context doesn't have enough info, output "I don't know".
#       3. Shows where the `context` and `question` will be placed when rendered.
#   - `from_template(...)` is a convenience constructor; there may also be `PromptTemplate(template=..., input_variables=[...])`
#     depending on the LangChain version.
#   - The template *string* is not yet the final prompt — it must be rendered with actual variable values.
#
# sys_prompt = sys_prompt_template.invoke({
#     "context": "The capital of India is New Delhi.",
#     "question": "What is the capital of France?"
# })
#   - `invoke(...)` on a PromptTemplate renders (fills) the template with the supplied variable mapping.
#       * It replaces `{context}` with "The capital of India is New Delhi."
#       * It replaces `{question}` with "What is the capital of France?"
#   - The result in `sys_prompt` is a single plain string containing the fully rendered prompt:
#       "Answer the following question using the given context.
#        If the context does not provide enough information, say 'I don't know'.
#        Context: The capital of India is New Delhi.
#        Question: What is the capital of France?"
#   - Note: Different LangChain versions sometimes name the rendering method `.format()` or `.format_prompt()` or `.apply()`.
#     Here `.invoke()` is used — it performs the rendering and returns the final prompt text.
#
# response = model.invoke(sys_prompt)
#   - Calls the model with the rendered prompt (synchronously in this code).
#   - `model.invoke(...)` sends the prompt string to the underlying API (OpenAI or other) and returns a response object.
#     The exact returned object type is wrapper-specific; common patterns:
#       * returns an object with `.content` or `.text` containing the model's text output
#       * may include metadata such as tokens used, finish_reason, raw API response, etc.
#   - Because the prompt asked to reply only if the context contains answerable info and to say "I don't know" otherwise,
#     the model should (ideally) answer "I don't know" here, because the context only mentioned India while the question asks about France.
#
# print(response.content)
#   - Prints the model's output content to stdout.
#   - `response.content` is the attribute used here to access the textual reply. Some wrappers use `response.text` or
#     `response.generations[0].text` — check the wrapper's docs for the correct attribute for your version.
#
# ---------------- Additional practical notes and cautions ----------------
# * API keys and authentication:
#     - Make sure your `.env` contains the required provider key (e.g., OPENAI_API_KEY or a provider-specific variable).
#     - If the wrapper reads a differently-named env var, check docs. Without a key, `model.invoke()` will raise an auth error.
#
# * Error handling:
#     - Real code should handle network errors, rate limits, invalid model names, and provider errors.
#     - Consider wrapping `model.invoke(...)` in try/except and checking response metadata if available.
#
# * Token and prompt length:
#     - Providers constrain prompt + completion tokens. If your prompt is long (large documents), you may hit limits.
#     - For RAG workflows you typically embed large contexts and pass only relevant chunks.
#
# * Determinism and temperature:
#     - With `temperature=0.0` the model will be more deterministic, but still may occasionally vary due to sampling settings.
#
# * Why the model should say "I don't know" in this example:
#     - Prompt instructs the model to use only the given `context`. The context mentions the capital of India.
#     - The question asks for the capital of France, which is not present in the context.
#     - The prompt tells the model explicitly to respond "I don't know" if context lacks info, so the expected output is that phrase.
#
# * Differences across LangChain versions:
#     - API names (module paths, method names like `.invoke()` vs `.generate()` vs `.call()`) can change between releases.
#     - If you get attribute errors, consult the version's docs or use introspection (`dir(model)`) to find the correct method.
#
# ---------------- Example of final rendered prompt (value of sys_prompt) ----------------
# Answer the following question using the given context.
# If the context does not provide enough information, say 'I don't know'.
# Context: The capital of India is New Delhi.
# Question: What is the capital of France?
#
# ------------------------------------------------------------------
# If you want, I can:
#  - Convert these comments into docstring-style comments at the top of the file,
#  - Add robust error handling and example `.env` content,
#  - Show a short unit-test or interactive example that demonstrates the expected "I don't know" output,
#  - Or update the code to use the latest LangChain calling pattern for your installed version.
# Which would you prefer? 😊
