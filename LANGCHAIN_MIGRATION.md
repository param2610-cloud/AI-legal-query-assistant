# LangChain Migration Guide
# AI Legal Assistant - Handling Deprecation Warnings

## Recent Changes Made (June 2025)

### 1. Import Updates
```python
# OLD (Deprecated)
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# NEW (Current)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
```

### 2. Agent Framework Updates
```python
# OLD (Deprecated)
from langchain.agents import initialize_agent

# NEW (Current)
from langchain.agents import create_react_agent, AgentExecutor
```

### 3. Chain Method Updates
```python
# OLD (Deprecated)
response = chain.run(query)
result = chain({"query": query})

# NEW (Current)
response = chain.invoke({"input": query})
result = chain.invoke({"query": query})
```

## Common Issues and Solutions

### Issue 1: Output Parsing Errors

**Problem**: `Could not parse LLM output` errors when using agents

**Solution**: 
- Set `handle_parsing_errors=True` in AgentExecutor
- Add `early_stopping_method="generate"` 
- Improve prompt templates with clearer format instructions

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    early_stopping_method="generate",
    max_iterations=3
)
```

### Issue 2: Chain Deprecation Warnings

**Problem**: `Chain.run` and `Chain.__call__` deprecated warnings

**Solution**: Replace with `invoke()` method
```python
# OLD
result = chain.run(query)
result = chain({"query": query})

# NEW
result = chain.invoke({"query": query})
result = chain.invoke({"input": query})
```

### Issue 3: Agent Initialization Changes

**Problem**: `initialize_agent` is deprecated

**Solution**: Use the new pattern:
```python
# OLD
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.REACT_DOCSTORE,
    verbose=True
)

# NEW
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

## Updated Dependencies

Make sure your requirements.txt includes:
```
langchain>=0.1.0
langchain-community>=0.0.10
langchain-ollama>=0.1.0
langchain-hub>=0.1.0
```

## Testing the Updates

Run your legal assistant to verify:
1. No more deprecation warnings
2. Agent responses work correctly
3. RAG system functions properly
4. Fallback mechanisms work when agent parsing fails

## Future-Proofing Tips

1. **Stay Updated**: Regularly check LangChain documentation
2. **Use Community Packages**: Import from `langchain_community` for third-party integrations
3. **Handle Errors Gracefully**: Always include error handling and fallbacks
4. **Test Thoroughly**: Verify agent responses after any updates

## Resources

- [LangChain Migration Guide](https://python.langchain.com/docs/versions/v0_2/)
- [AgentExecutor Migration](https://python.langchain.com/docs/how_to/migrate_agent/)
- [LangGraph ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)

## Current Status (June 2025)

✅ Imports updated to non-deprecated versions  
✅ Agent framework migrated to new pattern  
✅ Error handling improved with fallback mechanisms  
✅ Parsing errors handled gracefully  
✅ Output format standardized  

The system now works without deprecation warnings and includes robust error handling for better reliability.
