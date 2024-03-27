# Import necessary libraries
from pathlib import Path
import requests
import re
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    SummaryIndex,
)
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, load_index_from_storage, StorageContext
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.node_parser import SentenceSplitter

# Function Definitions

def setup_environment(api_key):
    """Sets up necessary environment variables."""
    os.environ["OPENAI_API_KEY"] = api_key

def fetch_and_save_wiki_articles(wiki_titles, data_path=Path("data")):
    data_path = Path("data")
    if not data_path.exists():
        data_path.mkdir()

    titles_with_no_extract = []

    # Assuming wiki_titles is defined somewhere above this snippet
    for title in wiki_titles:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        if 'extract' in page:
            wiki_text = page["extract"]
            # Apply sanitization when creating the file name
            safe_title = sanitize_title(title)
            with open(data_path / f"{safe_title}.txt", "w") as fp:
                fp.write(wiki_text)
        else:
            print(f"Extract not found for: {title}")
            titles_with_no_extract.append(title)

    # Remove titles with no extract from the original list
    for title in titles_with_no_extract:
        wiki_titles.remove(title)

    print("\nTitles with extracts successfully fetched and saved:")
    for title in wiki_titles:
        print(title)

def sanitize_title(title):
    """Sanitizes a title for safe file naming."""
    title = title.replace("(", "").replace(")", "")
    sanitized_title = re.sub(r"[^a-zA-Z0-9_ ]", "", title)
    sanitized_title = sanitized_title.replace(" ", "_")
    return sanitized_title[:64]

def load_wiki_docs(wiki_titles, data_path=Path("data")):
    """Loads wiki documents from saved files."""
    wiki_docs = {}
    for wiki_title in wiki_titles:
        wiki_docs[wiki_title] = SimpleDirectoryReader(
            input_files=[data_path / f"{wiki_title}.txt"]
        ).load_data()
    return wiki_docs

def setup_llama_indices(wiki_titles, wiki_docs):
    """Sets up LLaMA indices for wiki documents."""
    node_parser = SentenceSplitter()
    Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    agents = {}
    query_engines = {}
    for wiki_title in wiki_titles:
        nodes = node_parser.get_nodes_from_documents(wiki_docs[wiki_title])
        vector_index, summary_index = build_indices_for_title(wiki_title, nodes)
        agents[wiki_title], query_engines[wiki_title] = create_agent_and_query_engine(wiki_title, vector_index, summary_index)
    return agents, query_engines

def build_indices_for_title(wiki_title, nodes, data_path=Path("data")):
    """Builds or loads vector and summary indices for a given title."""
    storage_path = data_path / sanitize_title(wiki_title)
    if not storage_path.exists():
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=storage_path)
    else:
        vector_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=storage_path))
    summary_index = SummaryIndex(nodes)
    return vector_index, summary_index


def create_agent_and_query_engine(wiki_title, vector_index, summary_index):
    """Creates an OpenAIAgent and associated query engines for a given wiki title."""
    # Initialize query engines for vector and summary indices
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine()

    # Define tools for the agent to use with the query engines
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    "Useful for questions related to specific aspects of"
                    f" {wiki_title}. For detailed inquiries, technical discussions,"
                    " and precise information retrieval."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    "Ideal for comprehensive overviews or summaries of"
                    f" {wiki_title}. Use this for more general questions or when seeking"
                    " an understanding of broad topics."
                ),
            ),
        ),
    ]

    # Initialize the OpenAI model with specific settings for the agent
    function_llm = OpenAI(model="gpt-4")

    # Create the OpenAIAgent with the tools and language model
    agent = OpenAIAgent.from_tools(
        tools=query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about {wiki_title}.
Always utilize at least one tool when responding to a query to ensure accuracy and relevance.
Do not rely on general knowledge outside of provided tools and indices.\
""",
    )

    # Setup query engines for direct use, if needed, outside of agent interactions
    query_engines = {
        "vector_query_engine": vector_query_engine,
        "summary_query_engine": summary_query_engine,
    }

    return agent, query_engines

def define_tools_for_agents(agents):
    """Define and collect query engine tools for each document agent."""
    all_tools = []
    for wiki_title, agent in agents.items():
        wiki_summary = (
            f"This content contains Wikipedia articles about {wiki_title}. "
            "Use this tool if you want to answer any questions about {wiki_title}.\n"
        )
        doc_tool = QueryEngineTool(
            query_engine=agent,
            metadata=ToolMetadata(
                name=f"tool_{wiki_title}",
                description=wiki_summary,
            ),
        )
        all_tools.append(doc_tool)
    return all_tools

def create_object_index_and_retriever(all_tools):
    """Create an object index and retriever from the tools."""
    from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
    
    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
    obj_index = ObjectIndex.from_objects(
        all_tools, tool_mapping, VectorStoreIndex
    )
    return obj_index

def initialize_top_agent(obj_index):
    """Initialize the top-level agent designed to choose the appropriate document tool."""
    from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent
    
    top_agent = FnRetrieverOpenAIAgent.from_retriever(
        obj_index.as_retriever(similarity_top_k=3),
        system_prompt=(
            "You are an agent designed to answer queries about a wide range of topics. "
            "Please decide which document to use to answer the question. "
            "Always use the tools provided to answer a question. Do not rely on prior knowledge.\n"
        ),
        verbose=True,
    )
    return top_agent





def main():
    # Setup environment and API key
    api_key = "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"
    setup_environment(api_key)

    # Define or load your wiki titles
    wiki_titles = [
        "Healthcare in the United States",
        "Patient Protection and Affordable Care Act",
        "Medicaid",
        "Medicare (United States)",
        "Health insurance in the United States",
        "Health insurance marketplace",
        "Children's Health Insurance Program",
        "Health maintenance organization (HMO)",
        "Hospital readmission",
        "Electronic health record",
        "Telemedicine",
        "Mental health in the United States",
        "Opioid epidemic in the United States",
        "Centers for Disease Control and Prevention",
        "National Institutes of Health",
        "Food and Drug Administration",
        "Public health in the United States",
        "American Medical Association",
        "United States Department of Health and Human Services",
        "Health care reform in the United States",
        "Medical malpractice in the United States",
        "Health care prices in the United States",
        "COVID-19 pandemic in the United States",
        "UnitedHealth Group",
        "Anthem Inc.",
        "Aetna",
        "Cigna",
        "Humana",
        "Centene Corporation",
        "Molina Healthcare",
        "WellCare",
        "Blue Cross Blue Shield",
        "Kaiser Permanente",
        "Cerner",  # Added as suggested
        "Epic Systems",  # Adjusted for Wikipedia's naming, correct title is "Epic Systems"
        "Teladoc Health",  # Added as suggested
        "Amwell",  # Added as suggested
        "Preferred Provider Organization",  # Added as suggested, noting PPO may need specific article lookup
        "Personal health record"  # Adjusted as "Personal Health Records" might not directly match; use context to find correct title
    ]

    wiki_titles += [
        "Health Informatics",
        "Medical billing",
        "Health Insurance Portability and Accountability Act",
        "HL7",
        "International Classification of Diseases",
        "Current Procedural Terminology",
        "Electronic Data Interchange",
        "Healthcare Common Procedure Coding System",
        "Value-based health care",
        "Clinical audit"  # "Medical Audit" might be covered under "Clinical audit" based on Wikipedia's categorization.
    ]

    # Fetch and save articles, then load documents
    wiki_titles_with_extracts = fetch_and_save_wiki_articles(wiki_titles)
    wiki_titles = [sanitize_title(title) for title in wiki_titles]

    wiki_docs = load_wiki_docs(wiki_titles)

    # Setup LLaMA indices and create agents for individual documents
    agents, query_engines = setup_llama_indices(wiki_titles, wiki_docs)

    # Define tools for each document agent
    all_tools = define_tools_for_agents(agents)

    # Create an object index and retriever from the tools
    obj_index = create_object_index_and_retriever(all_tools)

    # Initialize the top-level agent with the object index
    top_agent = initialize_top_agent(obj_index)

    # Querying with the top-level agent
    response = top_agent.query("What does the FDA do, and what oversight do they have over the health industry?")
    print("Response from top agent:", response)

    # (Optional) Direct querying to specific agents or engines can be added here


if __name__ == "__main__":
    main()
