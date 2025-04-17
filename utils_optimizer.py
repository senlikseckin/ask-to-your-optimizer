from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value


def create_decision_variables():
    """Create decision variables for the expansion options."""
    # Decision variables are binary, indicating whether each option is chosen (1) or not (0).
    decision_vars = {
        "x_A": LpVariable("x_A", cat="Binary"),  # Option A: Expand existing facility
        "x_B": LpVariable("x_B", cat="Binary"),  # Option B: Build new facility
        "x_C": LpVariable("x_C", cat="Binary"),  # Option C: Implement new technology
    }
    return decision_vars


def define_parameters():
    """Define the parameters for the problem."""
    # Parameters include costs, capacity increases, labor requirements, and constraints.
    params = {
        "capital_costs": {
            "A": 1500000,
            "B": 2900000,
            "C": 1500000,
        },  # Capital costs for each option
        "operational_costs": {
            "A": 100000,
            "B": 200000,
            "C": 100000,
        },  # Monthly operational cost increases
        "labor_requirements": {
            "A": 15,
            "B": 30,
            "C": 15,
        },  # Additional skilled workers needed
        "additional_capacity": {
            "A": 40000,
            "B": 80000,
            "C": 40000,
        },  # Additional capacity per option
        "current_capacity": 100000,  # Current production capacity
        "budget": 35000000,  # Total budget for capital expenditure
        "max_labor_increase": 100,  # Maximum increase in skilled workers
        "labor_cost_per_worker": 5000,  # Monthly cost per additional skilled worker
        "demand": [
            170000,
            170000,
            170000,
            170000,
            170000,
        ],  # Projected demand for each year
    }
    return params


def setup_problem(decision_vars, params):
    """Set up the optimization problem."""
    # Create a minimization problem to minimize total costs.
    problem = LpProblem("Capacity_Expansion_Planning", LpMinimize)

    # Objective function: Minimize total cost over 5 years, including capital, operational, and labor costs.
    problem += (
        lpSum(params["capital_costs"][opt] * decision_vars[f"x_{opt}"] for opt in "ABC")
        + lpSum(
            (
                params["operational_costs"][opt]
                + params["labor_requirements"][opt] * params["labor_cost_per_worker"]
            )
            * decision_vars[f"x_{opt}"]
            for opt in "ABC"
        )
        * 12
        * 5  # 5 years
    )

    # Budget constraint: Ensure total capital expenditure does not exceed the budget.
    problem += (
        lpSum(params["capital_costs"][opt] * decision_vars[f"x_{opt}"] for opt in "ABC")
        <= params["budget"]
    )

    # Capacity constraints: Ensure total capacity meets or exceeds demand for each year.
    for t in range(5):
        problem += (
            params["current_capacity"]
            + lpSum(
                params["additional_capacity"][opt] * decision_vars[f"x_{opt}"]
                for opt in "ABC"
            )
            >= params["demand"][t]
        )

    # Labor constraint: Ensure the increase in skilled workers does not exceed the available labor market capacity.
    problem += (
        lpSum(
            params["labor_requirements"][opt] * decision_vars[f"x_{opt}"]
            for opt in "ABC"
        )
        <= params["max_labor_increase"]
    )

    return problem


def solve_problem(problem):
    """Solve the optimization problem and return the results."""
    # Solve the problem using the default solver.
    problem.solve(PULP_CBC_CMD(msg=False))

    # Extract the results, including the status and values of decision variables.
    results = {
        "status": LpStatus[problem.status],
        "x_A": problem.variablesDict()["x_A"].varValue,
        "x_B": problem.variablesDict()["x_B"].varValue,
        "x_C": problem.variablesDict()["x_C"].varValue,
        "total_cost": value(problem.objective),
    }
    return results


def main():
    # Create decision variables
    decision_vars = create_decision_variables()

    # Define parameters
    params = define_parameters()

    for key in params:
        print(key)
        print(params[key])
        print()

    capital_costs = {"A": 0}
    for key in capital_costs:
        params["capital_costs"][key] = capital_costs[key]

    for key in params:
        print(key)
        print(params[key])
        print()

    # Set up the problem
    problem = setup_problem(decision_vars, params)

    # Solve the problem
    results = solve_problem(problem)

    # Output the results
    print(f"Status: {results['status']}")
    print(f"Option A (Expand existing facility): {results['x_A']}")
    print(f"Option B (Build new facility): {results['x_B']}")
    print(f"Option C (Implement new technology): {results['x_C']}")
    print(f"Total Cost: ${results['total_cost']:,.2f}")


if __name__ == "__main__":
    main()
