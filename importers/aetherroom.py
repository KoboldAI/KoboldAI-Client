from dataclasses import dataclass
import requests
from typing import List

BASE_URL = "https://aetherroom.club/api/"


@dataclass
class ImportData:
    prompt: str
    memory: str
    authors_note: str
    notes: str
    title: str
    world_infos: List[object]


class RequestFailed(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__()


def import_scenario(id: int) -> ImportData:
    """
    Fetches story info from the provided AetherRoom scenario ID.
    """
    # Maybe it is a better to parse the NAI Scenario (if available), it has more data
    req = requests.get(f"{BASE_URL}{id}")
    if not req.ok:
        raise RequestFailed(req.status_code)

    json = req.json()
    prompt = json["promptContent"]
    memory = json["memory"]
    authors_note = json["authorsNote"]
    notes = json["description"]
    title = json.get("title", "Imported Story")

    world_infos = []
    for info in json["worldInfos"]:
        world_infos.append(
            {
                "key_list": info["keysList"],
                "keysecondary": [],
                "content": info["entry"],
                "comment": "",
                "folder": info.get("folder", None),
                "num": 0,
                "init": True,
                "selective": info.get("selective", False),
                "constant": info.get("constant", False),
                "uid": None,
            }
        )

    return ImportData(prompt, memory, authors_note, notes, title, world_infos)
