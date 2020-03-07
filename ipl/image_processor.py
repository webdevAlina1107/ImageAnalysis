from ipl.visualization import visualize_values_frequencies
import datetime as d


def main():
    seq = [1, 2, 3, 45]
    w = [50, 2, 10]
    e = {q: w for q, w in zip(seq, w)}
    visualize_values_frequencies(e)


if __name__ == '__main__':
    main()
