# Source Attribution:
# The majority of the code is derived from the following source:
# - OpenAttack GitHub Repository: https://github.com/thunlp/OpenAttack
# - Tag: v2.1.1
# - File: /OpenAttack/data/__init__.py
# - Reference Paper: OpenAttack: An Open-source Textual Adversarial Attack Toolkit (Zeng et al., ACL-IJCNLP 2021)
# Our modification is located between the comments "modification begin" and "modification end".

# modification begin
import os
# modification end
import pkgutil
import pickle
import urllib
from tqdm import tqdm
# modification begin
from OpenAttack.exceptions import DataConfigErrorException
from OpenAttack.data import data_list
from OpenAttack.data_manager import DataManager
# modification end

# modification begin
def add_data():
# modification end
    def pickle_loader(path):
        return pickle.load(open(path, "rb"))

    def url_downloader(url, resource_name):
        if url[0] == "/":
            remote_url = url[1:]
        else:
            remote_url = url

        def DOWNLOAD(path : str, source : str):
            CHUNK_SIZE = 1024
            if not source.endswith("/"):
                source = source + "/"
            with urllib.request.urlopen(source + remote_url) as fin:
                total_length = int(fin.headers["content-length"])
                with open(path, "wb") as fout:
                    with tqdm(total=total_length, unit="B", desc="Downloading %s" % resource_name, unit_scale=True) as pbar:
                        while True:
                            data = fin.read(CHUNK_SIZE)
                            if len(data) == 0:
                                break
                            fout.write(data)
                            pbar.update( len(data) )
            return True

        return DOWNLOAD

    for data in pkgutil.iter_modules(__path__):
        data = data.module_finder.find_loader(data.name)[0].load_module()
        if hasattr(data, "NAME") and hasattr(data, "DOWNLOAD"):  # is a data module
            tmp = {"name": data.NAME}
            if callable(data.DOWNLOAD):
                tmp["download"] = data.DOWNLOAD
            elif isinstance(data.DOWNLOAD, str):
                tmp["download"] = url_downloader(data.DOWNLOAD, data.NAME)
            else:
                raise DataConfigErrorException(
                    "Data Module: %s\n dir: %s\n type: %s"
                    % (data, dir(data), type(data.DOWNLOAD))
                )

            if hasattr(data, "LOAD"):
                tmp["load"] = data.LOAD
            else:
                tmp["load"] = pickle_loader
            # modification begin
            data_list.append(tmp)
            DataManager.AVAILABLE_DATAS.append(tmp["name"])
            DataManager.data_path[tmp["name"]] = os.path.join(os.getcwd(), "data", tmp["name"])
            DataManager.data_download[tmp["name"]] = tmp["download"]
            DataManager.data_loader[tmp["name"]] = tmp["load"]
            DataManager.data_reference[tmp["name"]] = None
            # modification end
        else:
            pass  # this is not a data module
