from wiki_fetcher import (setup_environment, fetch_and_save_wiki_articles, load_wiki_docs, 
                          setup_llama_indices, initialize_top_agent, sanitize_title, 
                          define_tools_for_agents, create_object_index_and_retriever)
from dotenv import load_dotenv
import os

load_dotenv()


# Setup environment and API key
api_key = os.getenv('API_KEY')

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


### EXAMPLE OF USE:
# Querying with the top-level agent

# response = top_agent.query("What does the FDA do, and what oversight do they have over the health industry?")
# print("Response from top agent:", response)



def fetch_response(query: str):
    """Fetches response for a given query using the top-level agent."""
    response = top_agent.query(query)
    return response