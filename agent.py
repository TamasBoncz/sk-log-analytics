import os
from dotenv import load_dotenv
import asyncio

# Import Semantic Kernel components
import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
)
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    DefaultTerminationStrategy,
)
from semantic_kernel.agents.strategies.selection.selection_strategy import (
    SelectionStrategy,
)
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)

from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt



# Import the ApiInteractionPlugin from the call_api_kernel_function.py file
from call_api_kernel_function import ApiInteractionPlugin

# Load environment variables from .env file
load_dotenv("/.env", override=True)
print("Environment set up successfully!")


def load_file(filename, directory=None):
    if directory:
        path = os.path.join(directory, filename)
    else:
        path = filename  # current root folder
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Load the log
logfile = load_file("logs.txt")

# Load instructions for each agent
log_interpreter_instructions = load_file("LogInterpreterAgent.txt", "agent-instructions")
research_agent_instructions = load_file("ResearchAgent.txt", "agent-instructions")
solution_synthesizer_agent_instructions = load_file("SolutionSynthesizerAgent.txt", "agent-instructions")
deep_dive_agent_instructions = load_file("DeepDiveAgent.txt", "agent-instructions")


def create_kernel_with_service(service_id):
    kernel = sk.Kernel()

    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
    )
    return kernel

# Create the kernel
kernel = create_kernel_with_service(service_id="multi-agent")
print("Kernel created successfully!")

# Add the utility plugin to the kernel
""" plugin = ApiInteractionPlugin()
kernel.add_plugin(plugin, plugin_name="ApiInteractionPlugin")
print(
    "ApiInteractionPlugin registered with functions:",
    [f for f in kernel.get_plugin("ApiInteractionPlugin").functions],
) """

settings = kernel.get_prompt_execution_settings_from_service_id(
    service_id="multi-agent"
)
# Configure automatic function calling
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()


LOG_INTERPRETER_AGENT_NAME = "log_interpreter_agent"
RESEARCH_AGENT_NAME = "research_agent"
SOLUTION_SYNTHETIZER_AGENT_NAME = "solution_synthetizer_agent"
DEEP_DIVE_AGENT_NAME = "deep_dive_agent"


# Log Interpreter Agent
log_interpreter_agent = ChatCompletionAgent(
    kernel=kernel,
    name=LOG_INTERPRETER_AGENT_NAME,
    instructions=log_interpreter_instructions,
    arguments=KernelArguments(
        settings=settings
    )
)
print(f"Agent '{log_interpreter_agent.name}' created successfully!")


# Research Agent
research_agent = ChatCompletionAgent(
    kernel=kernel,
    name=RESEARCH_AGENT_NAME,
    instructions=research_agent_instructions,
    arguments=KernelArguments(
        settings=settings
    )
)
print(f"Agent '{research_agent.name}' created successfully!")


# Solution Synthesizer Agent
solution_synthetizer_agent = ChatCompletionAgent(
    kernel=kernel,
    name=SOLUTION_SYNTHETIZER_AGENT_NAME,
    instructions=solution_synthesizer_agent_instructions,
    arguments=KernelArguments(
        settings=settings
    )
)
print(f"Agent '{solution_synthetizer_agent.name}' created successfully!")


# Deep Dive Agent
deep_dive_agent = ChatCompletionAgent(
    kernel=kernel,
    name=DEEP_DIVE_AGENT_NAME,
    instructions=deep_dive_agent_instructions,
    arguments=KernelArguments(
        settings=settings
    )
)
print(f"Agent '{deep_dive_agent.name}' created successfully!")




selection_function = KernelFunctionFromPrompt(
    function_name="selection",
    prompt=f"""
    Determine which participant takes the next turn in a conversation based on the the most recent participant.
    State only the name of the participant to take the next turn.
    No participant should take more than one turn in a row.

    Choose only from these participants:
    - {LOG_INTERPRETER_AGENT_NAME}
    - {RESEARCH_AGENT_NAME}
    - {SOLUTION_SYNTHETIZER_AGENT_NAME}
    - {DEEP_DIVE_AGENT_NAME}

    MAKE SURE TO ALWAYS FOLLOW THIS ORDER IN AGENT SELECTION:
        1. First the {LOG_INTERPRETER_AGENT_NAME}
        2. Second the {RESEARCH_AGENT_NAME}
        3. Third the {SOLUTION_SYNTHETIZER_AGENT_NAME} - MAKE SURE TO ALWAYS INVOKE THE {SOLUTION_SYNTHETIZER_AGENT_NAME} AFTER THE {RESEARCH_AGENT_NAME}

    EXCEPTION: IF the {SOLUTION_SYNTHETIZER_AGENT_NAME} or {DEEP_DIVE_AGENT_NAME} is the last that spoke, select the {DEEP_DIVE_AGENT_NAME}.

    History:
    {{{{$history}}}}
    """,
)

termination_function = KernelFunctionFromPrompt(
    function_name="termination",
    prompt="""
    Determine if the conversation is complete based on the last message.
    If the conversation is complete, answer with a single word and return "yes". Otherwise, return "no".

    If the {SOLUTION_SYNTHETIZER_AGENT_NAME} is the last terminate the conversation and answer "yes".
    If the {DEEP_DIVE_AGENT_NAME} is the last terminate the conversation and answer "yes".

    History:
    {{$history}}
    """,
)



log_analysis_group_chat = AgentGroupChat(
    agents=[log_interpreter_agent, research_agent, solution_synthetizer_agent, deep_dive_agent],
    selection_strategy=KernelFunctionSelectionStrategy(
        function=selection_function,
        kernel=kernel,
        result_parser=lambda result: str(result.value[0]) if result.value is not None else deep_dive_agent.name,
        agent_variable_name="agents",
        history_variable_name="history",
    ),
    termination_strategy=KernelFunctionTerminationStrategy(
        agents=[log_interpreter_agent, research_agent, solution_synthetizer_agent, deep_dive_agent],
        function=termination_function,
        kernel=kernel,
        result_parser=lambda result: str(result.value[0]).lower() == "yes",
        history_variable_name="history",
        maximum_iterations=3,
        # history_reducer=history_reducer,
    ),
)


async def run_group_chat(chat, user_message):
    """Run a multi-agent conversation and display the results.

    Args:
        chat: The AgentGroupChat instance
        user_message: The initial user message to start the conversation

    Returns:
        The chat history containing all messages
    """
    # Create a new chat history if needed
    if not hasattr(chat, "history") or chat.history is None:
        # Some versions of AgentGroupChat might not initialize history
        chat_history = ChatHistory()
        chat.history = chat_history

    # Add the user message to the chat
    await chat.add_chat_message(message=user_message)
    # print(f"\nUser: {user_message}\n")
    print("=== Beginning Agent Collaboration ===")

    # Track which agent is speaking for formatting
    current_agent = None

    # Invoke the chat and process agent responses
    try:
        async for response in chat.invoke():
            if response is not None and response.name:
                # Add a separator between different agents
                if current_agent != response.name:
                    current_agent = response.name
                    print(f"\n## {response.name}:\n{response.content}")
                else:
                    # Same agent continuing
                    print(f"{response.content}")

        print("\n=== Agent Collaboration Complete ===")
    except Exception as e:
        print(f"Error during chat invocation: {str(e)}")

    # Reset is_complete to allow for further conversations
    chat.is_complete = False

    return chat.history


question = """
        What is the solution to this problem?
        Analyse the provided logs and suggest a solution.
        The logs are as follows:
       """ + logfile


# Add the user message to the chat history
async def main():
    # Initial question
    await run_group_chat(log_analysis_group_chat, question)

    is_complete = False
    while not is_complete:
        print()
        user_input = input("User > ").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            is_complete = True
            print("[Exiting the conversation]")
            break

        if user_input.lower() == "reset":
            await log_analysis_group_chat.reset()
            print("[Conversation has been reset]")
            continue

        # Add the current user_input to the chat
        await log_analysis_group_chat.add_chat_message(message=user_input)

        await run_group_chat(log_analysis_group_chat, user_input)

        # Reset the chat's complete flag for the new conversation round.
        log_analysis_group_chat.is_complete = False

if __name__ == "__main__":
    asyncio.run(main())