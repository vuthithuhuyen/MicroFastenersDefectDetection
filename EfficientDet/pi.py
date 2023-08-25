import argparse
import math
import random


def chudnovsky(n):
    """
    Calculate Pi using the Chudnovsky algorithm.
    """
    pi = Decimal(0)
    for k in range(n):
        numerator = Decimal((-1) ** k) * math.factorial(6 * k) * (545140134 * k + 13591409)
        denominator = Decimal(math.factorial(3 * k)) * (math.factorial(k) ** 3) * (640320 ** (3 * k))
        pi += numerator / denominator
    pi = pi * Decimal(10005).sqrt() / 4270934400
    return 1 / pi


def monte_carlo_pi(num_points):
    """
    Calculate Pi using the Monte Carlo algorithm.
    """
    points_inside_circle = points_in_circle(num_points)
    return 4 * (points_inside_circle / num_points)


def points_in_circle(num_points):
    """
    Count the number of points that fall within the unit circle.
    """
    count_inside = 0
    for _ in range(num_points):
        if is_in_circle():
            count_inside += 1
    return count_inside


def is_in_circle():
    """
    Check if a point is within the unit circle.
    """
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    return x ** 2 + y ** 2 <= 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Pi using Chudnovsky or Monte Carlo algorithm.")
    parser.add_argument("--method", choices=["chudnovsky", "monte_carlo"], required=True,
                        help="Calculation method to use")
    parser.add_argument("--iterations", type=int, default=1000000, help="Number of iterations/points for calculation")
    args = parser.parse_args()

    if args.method == "chudnovsky":
        pi_value = chudnovsky(args.iterations)
    elif args.method == "monte_carlo":
        pi_value = monte_carlo_pi(args.iterations)

    print(f"Calculated Pi using {args.method} method with {args.iterations} iterations: {pi_value:.15f}")
