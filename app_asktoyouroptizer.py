# Import necessary libraries
import ast
import copy
import json

# Import ChromaDB and Streamlit
import chromadb
import streamlit as st

# Import dotenv for environment variable management
from dotenv import load_dotenv

# Import Langchain components
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings

# Import utility functions for optimization
from utils_optimizer import (
    create_decision_variables,
    define_parameters,
    setup_problem,
    solve_problem,
)

# Clear ChromaDB system cache
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Load environment variables
load_dotenv()


# Set the title of the Streamlit application
st.title(":giraffe_face: Ask to Your Optimizer")

# Load examples from the JSON file
with open("examples.json", "r") as file:
    examples = json.load(file)

# Join values of each dictionary in 'examples' into a single string
to_vectorize = [" ".join(example.values()) for example in examples]

# Create embeddings for text vectorization
embeddings = OpenAIEmbeddings()

# Store vector representations of text using Chroma
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# Select top 3 similar examples using semantic similarity
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=3,
)


def ask_to_optimizer(user_query, example_selector):
    """
    Function to interact with an optimizer model using a user query and example data.
    It processes the query, updates optimization parameters, and provides a summary of results.

    Parameters:
    - user_query: The query provided by the user to be processed by the optimizer.
    - example_selector: SemanticSimilarityExampleSelector class designed to find examples that are semantically similar to a given input.

    Returns:
    - Streamlit information message with the summarized results.
    """

    # Initialize the ChatOpenAI model with specified parameters
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # Define the prompt template for the chat model using example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),  # Template for human input
            ("ai", "{output}"),  # Template for AI output
        ]
    )

    # Create a few-shot prompt template using the example selector and example prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        example_selector=example_selector,
        example_prompt=example_prompt,
    )

    # Define the system message for the AI assistant
    system_message = """
        As an AI assistant, you are equipped with a knowledge base of sample
        questions and answers.

        Your task is to interpret the user's message, understand the parameter
        they wish to modify, and respond with the appropriate answer.

        Please avoid including any reasoning in your output. Your response should
        solely consist of the output examples in example questions answer pairs.
    """

    # Create the final prompt template for the chat model
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    # Combine the final prompt with the model to create a processing chain
    chain = final_prompt | model

    # Invoke the chain with the user query to get the response
    res = chain.invoke({"input": user_query})

    # Extract the content to pass to the optimizer
    pass_to_optimizer = res.content

    # Set up the optimization problem with initial parameters
    decision_vars = create_decision_variables()
    params = define_parameters()
    problem = setup_problem(decision_vars, params)
    por_results = solve_problem(problem)

    original_params = copy.deepcopy(params)

    # Parse the updates from the AI response
    updates = ast.literal_eval(pass_to_optimizer)

    params_list = [
        "capital_costs",
        "operational_costs",
        "labor_requirements",
    ]  # parameter lists included in example QA database

    # Check if the response contains a message indicating no updates
    if list(updates.keys())[0] not in params_list:
        return st.info("Please ask your question")
    else:
        # Update the parameters based on the AI response
        for key1 in updates:
            for key2 in updates[key1]:
                params[key1][key2] = updates[key1][key2]

    # Re-solve the optimization problem with updated parameters

    problem = setup_problem(decision_vars, params)
    wif_results = solve_problem(problem)

    differences = {}

    for key in original_params:
        value1 = original_params[key]
        value2 = params[key]

        if isinstance(value1, list) and isinstance(value2, list):
            if value1 != value2:
                differences[key] = {"Original": value1, "WIF": value2}
        else:
            if value1 != value2:
                differences[key] = {"Original": value1, "WIF": value2}

    for key in differences:
        diffence_summary = (
            f"Updated parameter based on user input: {key}\n"
            + f"Original Scenario Values: {differences[key]['Original']}\n"
            + f"WIF Scenario Values: {differences[key]['WIF']}\n"
        )

    # Generate a summary of the por optimization results
    por_summary = (
        f"Status: {por_results['status']}\n"
        + f"Option A (Expand existing facility): {por_results['x_A']}\n"
        + f"Option B (Build new facility): {por_results['x_B']}\n"
        + f"Option C (Implement new technology): {por_results['x_C']}\n"
        + f"Total Cost: ${por_results['total_cost']:,.2f}\n"
    )

    # Generate a summary of the updated optimization results
    wif_summary = (
        f"Status: {wif_results['status']}\n"
        + f"Option A (Expand existing facility): {wif_results['x_A']}\n"
        + f"Option B (Build new facility): {wif_results['x_B']}\n"
        + f"Option C (Implement new technology): {wif_results['x_C']}\n"
        + f"Total Cost: ${wif_results['total_cost']:,.2f}\n"
    )

    # Define the system message for summarizing results
    system_message_summarize = """
        As an AI assistant, you will summarize the differences and similarities between two analysis, POR (Plan of Record) and WIF(What-IF).
        First, explain what user asked for.
        Second, explain which parameter is being updated, what was the original value and WIF scenario value based on user input
        Third, include details of both analysis in your response.
        Fourth, explain the differences and similarities between two analysis
        Lastly, make recommendations on which option should be selected.
    """

    # Create the final prompt template for summarizing results
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_summarize),
            ("human", "{input}"),
        ]
    )

    # Combine the original and updated summaries for comparison
    total_summary = (
        "User Request\n:"
        + user_query
        + "\n"
        + "Difference Summary\n\n"
        + diffence_summary
        + "\n POR Analysis\n\n"
        + por_summary
        + "\n WIF Analysis\n\n"
        + wif_summary
    )

    # Create a processing chain for summarizing results
    chain = final_prompt | model
    res = chain.invoke({"input": total_summary})

    # Return the summarized results as a Streamlit information message
    return st.info(res.content)


# Create a form with a text input and submit button
with st.form("myform"):
    # Input field for user to enter a prompt
    user_query = st.text_input("Enter prompt:", "")
    # Button to submit the form
    submitted = st.form_submit_button("Submit")
    # Call function with user input and example selector
    ask_to_optimizer(user_query, example_selector)
