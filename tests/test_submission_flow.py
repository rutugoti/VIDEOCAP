import os
import json
import pytest
from unittest.mock import MagicMock, patch
from main import run_tasks_json_pipeline, download_video
from src.shared.models import Caption


def test_download_video_failure():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = Exception("Connection error")
        with pytest.raises(IOError, match="Failed to download video"):
            download_video("http://example.com/video.mp4")


def test_run_tasks_json_pipeline(tmp_path):
    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"
    
    # Create input tasks.json content
    tasks_content = [
        {
            "task_id": "task_test_1",
            "video_url": "http://example.com/test_video.mp4",
            "styles": ["formal", "humorous_tech"]
        }
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(tasks_content, f)

    # Mock the orchestrator
    mock_orchestrator = MagicMock()
    # Mock return value of process_video (dict of styles to Caption objects)
    mock_orchestrator.process_video.return_value = {
        "formal": Caption(text="Formal caption text", style="formal", word_count=3),
        "tech_humor": Caption(text="Tech humor caption text", style="tech_humor", word_count=4),
    }

    # Patch download_video and os.remove
    with patch("main.download_video") as mock_download, \
         patch("os.remove") as mock_remove:
        local_temp_file = str(tmp_path / "local_video.mp4")
        mock_download.return_value = local_temp_file
        # Create empty file so os.path.exists is True and cleanup runs
        with open(local_temp_file, "w") as f:
            f.write("")
        
        # Run submission flow
        run_tasks_json_pipeline(str(input_file), str(output_file), mock_orchestrator)
        
        # Assertions
        mock_download.assert_called_once_with("http://example.com/test_video.mp4")
        mock_orchestrator.process_video.assert_called_once_with(local_temp_file)
        mock_remove.assert_any_call(local_temp_file)
        
        # Verify results.json content and schema
        assert os.path.exists(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            
        assert len(results) == 1
        assert results[0]["task_id"] == "task_test_1"
        assert "captions" in results[0]
        # Only requested styles should be in the output
        assert results[0]["captions"] == {
            "formal": "Formal caption text",
            "humorous_tech": "Tech humor caption text"
        }
