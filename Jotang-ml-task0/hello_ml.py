import numpy as np

def calculate_average(scores):
    if not scores:
        raise ValueError("Scores list is empty")
    return sum(scores) / len(scores)

def find_highest_score(students):
    if not students:
        raise ValueError("Students list is empty")
    top_name = max(students, key=lambda name: students[name])
    return top_name, students[top_name]

def main():
    students = {
        "Alice": 90,
        "Bob": 85,
        "Charlie": 95,
        "David": 88,
    }
    average_score = calculate_average(students.values())
    top_name, top_score = find_highest_score(students)
    print(f"学生成绩：{students}")
    print(f"平均成绩：{average_score:.2f}")
    print(f"最高成绩：{top_name},{top_score}")

    matrix_a = np.array([
        [1,2,3],
        [4,5,6],
    ])
    matrix_b = np.array([
        [1,0],
        [0,1],
        [1,1],
    ])
    matrix_c = np.dot(matrix_a, matrix_b)
    print(f"\n矩阵 A:\n{matrix_a}")
    print(f"A 的形状: {matrix_a.shape}")
    print(f"\n矩阵 B:\n{matrix_b}")
    print(f"B 的形状: {matrix_b.shape}")
    print(f"\n矩阵 C = A @ B:\n{matrix_c}")
    print(f"C 的形状: {matrix_c.shape}")

if __name__ == "__main__":
    main()
