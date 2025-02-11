import sqlite3
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal
import app.cortex._schemas as schemas
from langchain.tools import Tool
from ._utils import utils
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import AIMessage
from app.data_layer.services.conversation_service import ConversationService
from app.data_layer.models.conversation import Message
from ..core.reasoner.resoning_engine import ReasoningEngine
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_executor_for_config,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from app.logging_config import agent_logger
from app.config import gemini_pro_model_langchain
from .tools import table_operator

llm_model = gemini_pro_model_langchain

# Define a prompt that asks the model to decide if a retrieval is needed.
decision_system_prompt = """Given the following conversation history and the new user question, decide whether to:
	1.	Perform an internal knowledge search (RAG) if the question requires retrieving information beyond the conversation context.
	2.	Call the table operator tool if the user references an existing table and requests modifications.
	3.	Respond directly if the question can be answered without additional retrieval or table modifications.

Execute the appropriate tool call accordingly."""


decision_prompt = ChatPromptTemplate.from_messages([
    ("system", decision_system_prompt),
    ("placeholder", "{messages}"),
    # ("human", "{input}")
])

# Define your external retrieval tool
def internal_knowledge_search(query: str) -> str:
    # Your retrieval logic here (e.g., search a vectorstore or an API call)
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    reasoning = ReasoningEngine(username=configurable["user_id"], query=query)
    response = reasoning.start_reasoning()
    conversation_service = ConversationService()
    reasoning_message = Message(
        sender="cortex",
        type="internal_knowledge",
        content=response.final_answer,
        reasoning=response.reasoning,
        table=response.table
    )
    conversation_service.store_message(
        message=reasoning_message,
        user_id=configurable["user_id"],
        conversation_id=configurable["thread_id"]
    )
    return reasoning_message

def format_thought_output(thought: Message) -> str:
    """
    Formats a ThoughtOutput (or Message) object to include its final_answer and table for LLM input.
    """
    formatted = f"Answer: {thought.content}"
    if thought.table:
        formatted += f"\nTable:\n{thought.table}"
    return formatted

internal_search_tool = Tool(
    name="KnowledgeSearch",
    func=internal_knowledge_search,
    description="Searches internal knowledge base for information relevant to the user's query."
)

all_tools = [internal_search_tool , table_operator]

def remove_empty_messages(messages):
    return [msg for msg in messages if msg.content != '']

def decide_route(state, config: dict):
    # Use an LLM prompt to decide:
    llm = utils.get_llm(llm_model)
    # formatted_prompt = decision_prompt.format(messages=state['messages'], input=state['new_query'])
    # agent_logger.info(f"Agent prompt:\n {formatted_prompt}")
    bound = decision_prompt | llm.bind_tools(all_tools)
    
    agent_logger.info(f"Agent thread_id at start:\n {config['configurable']['thread_id']}")
    # agent_logger.info(f"Agent state new_query:\n {state['new_query']}")
    msg = state["messages"][-1]
    state['messages'] = state['messages'][:4]
    state['messages'].append(msg)
    # state['messages'] = remove_empty_messages(messages)

    agent_logger.info(f"Agent state messages:\n {state['messages']}")
    
    decision = bound.invoke( {
            "messages": state['messages'],
            # "input": state['new_query']
        })
    # state["messages"].append(state["new_query"])
    state["messages"].append(decision)
    return state

def route_tools(state: schemas.State, config: dict):
    """Determine whether to call a tool and update state directly instead of routing to a ToolNode."""
    # Get the last message produced by decide_route.
    msg = state["messages"][-1]
    agent_logger.info(f"Agent response to new user query:\n {msg}")
    
    # Check if the message indicates a tool call.
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        state["messages"].pop()
        # Extract the first tool call from the list.
        tool_call = msg.tool_calls[0]
        # Access the tool call's attributes directly.
        tool_name = tool_call.get("name")
        tool_input = tool_call.get("args")

        # If the tool is "KnowledgeSearch", call it directly.
        if tool_name == "KnowledgeSearch":
            # Directly invoke the tool function.
            tool_output = internal_knowledge_search(tool_input)
            
        # If the tool is "TableOperator", call it directly.
        if tool_name == "TableOperator":
            # Directly invoke the tool function.
            tool_output = table_operator(tool_input)
        
        formatted_output = format_thought_output(tool_output)
        # Update your state with the tool's output.
        state["messages"].append(
            AIMessage(content=formatted_output)
        )
        state["output"] = tool_output
            
    else : 
        # After processing the tool call, return END to signal completion.
        response = msg.content
        conversation_service = ConversationService()
        reasoning_message = Message(
            sender="cortex",
            type="from_conversation",
            content=response,
        )
        conversation_service.store_message(
            message=reasoning_message,
            user_id=config["configurable"]["user_id"],
            conversation_id=config["configurable"]["thread_id"]
        )
        state["output"] = reasoning_message
    
    agent_logger.info(f"Agent thread_id at end:\n {config['configurable']['thread_id']}")
    agent_logger.info(f"Agent state at end:\n {state}")
    return state

# Create the graph and add nodes
builder = StateGraph(schemas.State, schemas.GraphConfig)
builder.add_node("decide", decide_route)
builder.add_node("call_tool", route_tools)
# Add edges to the graph
builder.add_edge(START, "decide")
builder.add_edge("decide", "call_tool")
builder.add_edge("call_tool", END)

conn = sqlite3.connect('checkpoints.sqlite', check_same_thread=False)
memory = SqliteSaver(conn)

cortex = builder.compile(checkpointer=memory)
                         
__all__ = ["cortex"]