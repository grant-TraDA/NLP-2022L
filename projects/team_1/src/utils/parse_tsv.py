import sys
from typing import List, Tuple


def __read_times_data(path: str) -> List[Tuple]:
    data = []
    for line in open(path, encoding="utf-8"):
        if line[-1] == "\n":
            line = line[:-1]
        if line == "</s>":
            continue
        times, text = line.split(" ")
        start, end = times[1:-1].split(",")
        start = int(start)
        end = int(end)

        data.append((start, end, text))
    return data


def __times_after_tokens(path: str) -> List[Tuple]:
    data = __read_times_data(path)
    result = []
    for i in range(len(data)):
        start, end, text = data[i]

        try:
            br = data[i + 1][0] - end
        except IndexError:
            br = 0
        result.append((text, br))
    return result


def __match_times(times: Tuple, expected: List) -> List:
    matched = []

    times_text = ""
    times_indexes = {}
    for token, time in times:
        times_indexes[len(times_text)] = time
        times_text = token.lower()

    index = 0
    for token in expected:
        found_index = times_text.find(token.lower(), index)
        if found_index >= 0:
            if found_index in times_indexes:
                matched.append(times_indexes[found_index])
                index = found_index
            else:
                matched.append(-1)
        else:
            matched.append(-1)
    return matched


def parse_tsv(
    in_path: str,
    expected_path: str,
    save_path: str,
    clntmstmp_dir: str = None,
    files_to_ignore: List = []
) -> None:
    out = open(save_path, "w", encoding="utf-8")

    for in_line, expected in zip(
        open(in_path, encoding="utf-8"),
        open(expected_path, encoding="utf-8")
    ):
        if in_line[-1] == "\n":
            in_line = in_line[:-1]
        name, text = in_line.split("\t")

        if name in files_to_ignore:
            continue

        if expected[-1] == "\n":
            expected = expected[:-1]

        assert len(text.split(" ")) == len(expected.split(" "))

        if clntmstmp_dir:
            times = __times_after_tokens(clntmstmp_dir + f"/{name}.clntmstmp")
            matched = __match_times(times, expected.split(" "))

        for i, (in_token, expected_token) in enumerate(
            zip(text.split(" "), expected.split(" "))
        ):
            expected_token = expected_token.lower()
            if in_token == expected_token:
                label = "B"
            else:
                if in_token == expected_token[:-1]:
                    label = expected_token[-1]
                elif in_token == expected_token[:len(in_token)]:
                    label = expected_token[len(in_token)]
                else:
                    print("ERROR", in_token, expected_token, file=sys.stderr)

            if clntmstmp_dir:
                out.write(f"{in_token}\t{label}\t{matched[i]}\n")
            else:
                out.write(f"{in_token}\t{label}\n")
        out.write("\n")
