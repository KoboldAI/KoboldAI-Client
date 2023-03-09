import pytest
import requests_mock

from importers.aetherroom import (
    ImportData,
    RequestFailed,
    import_scenario,
)


def test_import_scenario_http_error(requests_mock: requests_mock.mocker):
    requests_mock.get("https://aetherroom.club/api/1", status_code=404)
    with pytest.raises(RequestFailed):
        import_scenario(1)


def test_import_scenario_success(requests_mock: requests_mock.Mocker):
    json = {
        "promptContent": "promptContent",
        "memory": "memory",
        "authorsNote": "authorsNote",
        "description": "description",
        "title": "title",
        "worldInfos": [],
    }
    requests_mock.get("https://aetherroom.club/api/1", json=json)

    expected_import_data = ImportData(
        "promptContent", "memory", "authorsNote", "description", "title", []
    )
    assert import_scenario(1) == expected_import_data


def test_import_scenario_no_title(requests_mock: requests_mock.Mocker):
    json = {
        "promptContent": "promptContent",
        "memory": "memory",
        "authorsNote": "authorsNote",
        "description": "description",
        "worldInfos": [],
    }
    requests_mock.get("https://aetherroom.club/api/1", json=json)

    expected_import_data = ImportData(
        "promptContent", "memory", "authorsNote", "description", "Imported Story", []
    )
    assert import_scenario(1) == expected_import_data


def test_import_scenario_world_infos(requests_mock: requests_mock.Mocker):
    json = {
        "promptContent": "promptContent",
        "memory": "memory",
        "authorsNote": "authorsNote",
        "description": "description",
        "worldInfos": [
            {
                "entry": "Info 1",
                "keysList": ["a", "b", "c"],
                "folder": "folder",
                "selective": True,
                "constant": True,
            },
            {
                "entry": "Info 2",
                "keysList": ["d", "e", "f"],
                "folder": "folder 2",
                "selective": True,
                "constant": True,
            },
        ],
    }
    requests_mock.get("https://aetherroom.club/api/1", json=json)

    expected_import_data = ImportData(
        "promptContent",
        "memory",
        "authorsNote",
        "description",
        "Imported Story",
        [
            {
                "content": "Info 1",
                "key_list": ["a", "b", "c"],
                "keysecondary": [],
                "comment": "",
                "num": 0,
                "init": True,
                "uid": None,
                "folder": "folder",
                "selective": True,
                "constant": True,
            },
            {
                "content": "Info 2",
                "key_list": ["d", "e", "f"],
                "keysecondary": [],
                "comment": "",
                "num": 0,
                "init": True,
                "uid": None,
                "folder": "folder 2",
                "selective": True,
                "constant": True,
            },
        ],
    )
    assert import_scenario(1) == expected_import_data


def test_import_scenario_world_info_missing_properties(
    requests_mock: requests_mock.Mocker,
):
    json = {
        "promptContent": "promptContent",
        "memory": "memory",
        "authorsNote": "authorsNote",
        "description": "description",
        "worldInfos": [
            {
                "entry": "Info 1",
                "keysList": ["a", "b", "c"],
            }
        ],
    }
    requests_mock.get("https://aetherroom.club/api/1", json=json)

    expected_import_data = ImportData(
        "promptContent",
        "memory",
        "authorsNote",
        "description",
        "Imported Story",
        [
            {
                "content": "Info 1",
                "key_list": ["a", "b", "c"],
                "keysecondary": [],
                "comment": "",
                "num": 0,
                "init": True,
                "uid": None,
                "folder": None,
                "selective": False,
                "constant": False,
            }
        ],
    )
    assert import_scenario(1) == expected_import_data
