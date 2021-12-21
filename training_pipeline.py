from training.create_model import initialize
import hashlib
import os
from uuid import uuid4
import json
import Algorithmia
from time import time
import sys


def hash_directory(path):
    digest = hashlib.sha1()

    for root, dirs, files in os.walk(path):
        for names in files:
            file_path = os.path.join(root, names)

            # Hash the path and add to the digest to account for empty files/directories
            digest.update(hashlib.sha1(file_path[len(path):].encode()).digest())

            # Per @pt12lol - if the goal is uniqueness over repeatability, this is an alternative method using 'hash'
            # digest.update(str(hash(file_path[len(path):])).encode())

            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f_obj:
                    while True:
                        buf = f_obj.read(1024 * 1024)
                        if not buf:
                            break
                        digest.update(buf)

    return digest.hexdigest()


def process_cifar10(client):
    model_name = "cifar10-" + str(uuid4()) + ".t7"
    manifest_model_name = "cifar10"
    data_collection = "data://algorithmia_admin/cifar10_models"
    remote_filepath = f"{data_collection}/{model_name}"
    local_location = "/tmp/cifar10_data"
    data = initialize(model_name, local_location)
    dataset_hash = hash_directory(local_location)
    client.file(remote_filepath).putFile(data['filepath'])
    with open('model_manifest.json', "r") as f:
        man_data = json.load(f)
    found = False
    for i in range(len(man_data['required_files'])):
        if man_data['required_files'][i]['name'] == manifest_model_name:
            found = True
            man_data['required_files'][i]['source_uri'] = remote_filepath
            man_data['required_files'][i]['metadata'] = {'dataset_md5_checksum': dataset_hash}
            man_data['required_files'][i]['date_modified'] = str(time())
    if not found:
        model = {'name': manifest_model_name, 'source_uri': remote_filepath,
                 'metadata': {'dataset_md5_checksum': dataset_hash, 'date_modified': str(time())}}
        man_data['required_files'].append(model)
    with open('model_manifest.json', "w") as f:
        json.dump(man_data, f)


if __name__ == "__main__":
    model = str(sys.argv[1])
    client = Algorithmia.client()
    if model == "cifar10":
        process_cifar10(client)
    else:
        raise Exception(f"Model type: {model} not supported")
    client.freeze()
