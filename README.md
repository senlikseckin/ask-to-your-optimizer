AI Agent-Powered Decision Making: Ask To Your Optimizer
=================================

Welcome to the Ask to Your Optimizer! This initiative utilizes agentic flows to help users assess various scenarios for a company aiming to create a strategic plan to expand production capacity over the next five years. The goal is to meet rising demand while minimizing costs and maintaining service quality.

This project showcases how AI agents can streamline and expedite the decision-making process. By allowing decision-makers to interact directly with an optimization engine, they can explore "what if" scenarios with an AI agentic flow and receive instant feedback. This method removes the need for experts like planners, analysts, and engineers, thus speeding up and enhancing decision-making efficiency.

This project establishes a platform for users to engage with the problem-solving tool through conversation.

# Problem Statement

Peter Pan Smart Solutions, located in Neverland, specializes in producing high-end smart home devices and has demonstrated strong sales performance. The business problem they face involves developing a strategic plan to expand production capacity over the next five years to accommodate increasing demand, while simultaneously minimizing costs and maintaining service quality.

# Architecture

The first component involves defining the business problem and creating an optimization solution based on specific inputs and constraints.utils_optimizer.py file has the ingredients of the problem stament and solution.

The second component is an interactive process where users can ask questions in plain English - no coding, no prior knowledge on details of optimization code needed. Based on these questions, parameters such as the capital cost of different options are updated. These updated parameters are then fed into the optimizer, which runs alongside the original parameters to provide comparisons and recommendations to the user.The app_asktoyouroptizer.py file contains the key components for the agentic flow. The examples.json file includes sample question-and-answer pairs for few-shot learning utilized in the flow.

Users can ask "what if" scenarios by asking questions through a web UI, and the app returns the final answer and recommendation. The video below demonstrates how the app works.

# Demo

[![YouTube](http://i.ytimg.com/vi/SsFVzYiwx3I/hqdefault.jpg)](https://www.youtube.com/watch?v=SsFVzYiwx3I)

https://www.youtube.com/watch?v=SsFVzYiwx3I_channel=SeckinSenlik



# Installation

To get started with the AI-Powered Music Festival Planner, follow these steps:

1. git clone https://github.com/senlikseckin/ask-to-your-optimizer.git
2. cd ask-to-your-optimizer
3. pip install -r requirements.txt
4. Enter your personal OpenAPI key in .env file


# Usage

To launch the user interface through the terminal and start chatting with the AI Agents, use the
following command:

``` sh
    $ streamlit run app_asktoyouroptimizer.py
```
