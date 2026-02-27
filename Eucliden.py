import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)))

def main():
    try:
        print("Enter the coordinates of two points:")

        point1 = list(map(float, input(
            "Enter coordinates of Point 1 (space-separated): "
        ).split()))

        point2 = list(map(float, input(
            "Enter coordinates of Point 2 (space-separated): "
        ).split()))

        if len(point1) != len(point2):
            print("Error: Points must have the same number of dimensions.")
            return

        euclidean = euclidean_distance(point1, point2)
        manhattan = manhattan_distance(point1, point2)

        print(f"Euclidean Distance: {euclidean:.4f}")
        print(f"Manhattan Distance: {manhattan:.4f}")

    except ValueError:
        print("Error: Please enter valid numeric values separated by spaces.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()