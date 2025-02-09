import math

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def calculate_area(radius):
    return math.pi * radius * radius

def example_function(x):
    # This is a comment that might be too long and subject to optimization
    # This function calculates the sine of the input and returns it
    y = math.sin(x)
    return y

def generate_random_numbers(count):
    numbers = []
    for _ in range(count):
        numbers.append(random.randint(1, 100))
    return numbers

def main():
    radius = 5
    area = calculate_area(radius)
    print(f"Area of the circle: {area}")

    x = 3.14
    sine_value = example_function(x)
    print(f"Sine of {x}: {sine_value}")

    num = 29
    if is_prime(num):
        print(f"{num} is a prime number")
    else:
        print(f"{num} is not a prime number")

    random_numbers = generate_random_numbers(10)
    print(f"Random numbers: {random_numbers}")

if __name__ == "__main__":
    main()