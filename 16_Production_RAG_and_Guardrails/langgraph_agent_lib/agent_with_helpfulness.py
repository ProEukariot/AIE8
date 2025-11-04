"""An agent graph with a post-response helpfulness check loop.

After the agent responds, a secondary node evaluates helpfulness ('Y'/'N').
If helpful, end; otherwise, continue the loop or terminate after a safe limit.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict, Annotated

from .models import get_openai_model
from .rag import ProductionRAGChain
from .caching import setup_llm_cache
from .agents import get_default_tools


class AgentState(TypedDict):
    """State schema for agent graphs, storing a message list with add_messages."""
    messages: Annotated[List[BaseMessage], add_messages]


def _build_model_with_tools(
    model_name: str = "gpt-4.1-nano",
    temperature: float = 0,
    tools: Optional[List] = None
):
    """Return a chat model instance bound to the current tool belt."""
    model = get_openai_model(model_name=model_name, temperature=temperature)
    return model.bind_tools(tools or [])


def call_model(state: AgentState, model_name: str = "gpt-4.1-nano", temperature: float = 0, tools: Optional[List] = None) -> Dict[str, Any]:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools(model_name=model_name, temperature=temperature, tools=tools)
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_helpfulness(state: AgentState):
    """Decide whether to execute tools or run the helpfulness evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "helpfulness"


def helpfulness_node(
    state: AgentState,
    helpfulness_model_name: str = "gpt-4.1-mini"
) -> Dict[str, Any]:
    """Evaluate helpfulness of the latest response relative to the initial query."""
    # If we've exceeded loop limit, short-circuit with END decision marker
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="HELPFULNESS:END")]}    

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    prompt_template = """Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

Initial Query:
{initial_query}

Final Response:
{final_response}"""

    helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
    helpfulness_check_model = get_openai_model(model_name=helpfulness_model_name)
    helpfulness_chain = (
        helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()
    )

    helpfulness_response = helpfulness_chain.invoke(
        {
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        }
    )

    decision = "Y" if "Y" in helpfulness_response else "N"
    return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}


def helpfulness_decision(state: AgentState):
    """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
    # Check loop-limit marker
    if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if "HELPFULNESS:Y" in text:
        return "end"
    return "continue"


def build_graph(
    model_name: str = "gpt-4.1-nano",
    temperature: float = 0,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    helpfulness_model_name: str = "gpt-4.1-mini",
    cache_type: str = "sqlite",
    cache_path: Optional[str] = None
):
    """Build an agent graph with an auxiliary helpfulness evaluation subgraph.
    
    Args:
        model_name: OpenAI model name for the main agent
        temperature: Model temperature
        tools: Optional list of tools. If None, uses get_default_tools
        rag_chain: Optional RAG chain to include as a tool
        helpfulness_model_name: Model name for helpfulness evaluation
        cache_type: Type of cache - "memory" or "sqlite"
        cache_path: Path for SQLite cache file
        
    Returns:
        Compiled LangGraph agent
    """
    # Set up LLM caching for all model calls
    setup_llm_cache(cache_type=cache_type, cache_path=cache_path)
    
    # Get tools if not provided
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Create closure to capture tools and model config
    def call_model_node(state: AgentState) -> Dict[str, Any]:
        return call_model(state, model_name=model_name, temperature=temperature, tools=tools)
    
    def helpfulness_node_wrapper(state: AgentState) -> Dict[str, Any]:
        return helpfulness_node(state, helpfulness_model_name=helpfulness_model_name)
    
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    graph.add_node("agent", call_model_node)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node_wrapper)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph.compile()


# Default graph instance (can be overridden with custom parameters)
graph = build_graph()
