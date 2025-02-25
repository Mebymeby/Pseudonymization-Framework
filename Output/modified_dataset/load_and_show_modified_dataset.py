import json
import os
import datasets

if __name__ == "__main__":

    dataset_name = "xsum"
    dataset_type = "only_rep"  # only_rep tag_wrap
    dataset_path = os.path.join("Output", "modified_dataset", dataset_name, dataset_type)

    data = datasets.load_from_disk(dataset_path)
    print(data.shape)

    data = data.select(range(10))
    for item in data:
        print(json.dumps(item, indent=4, ensure_ascii=False))
