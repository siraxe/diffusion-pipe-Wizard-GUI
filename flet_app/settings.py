import json
import os
import re


class Config:
    _instance = None
    _settings = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_settings()
        return cls._instance

    def _load_settings(self):
        settings_file_path = os.path.join(os.path.dirname(__file__), "settings.json")
        try:
            with open(settings_file_path, "r", encoding="utf-8") as f:
                self._settings = json.load(f)
            self._post_process_settings()
        except FileNotFoundError:
            print(f"Error: settings.json not found at {settings_file_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {settings_file_path}. Check for syntax errors.")

    @staticmethod
    def _running_in_wsl() -> bool:
        if os.name != "posix":
            return False
        if "WSL_DISTRO_NAME" in os.environ or "WSL_INTEROP" in os.environ:
            return True
        try:
            with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as f:
                return "microsoft" in f.read().lower()
        except Exception:
            return False

    @classmethod
    def _normalize_path_for_platform(cls, raw_path: str) -> str:
        if not raw_path:
            return raw_path

        path_val = str(raw_path)

        if cls._running_in_wsl():
            match = re.match(r"^[A-Za-z]:[\\/](.*)", path_val)
            if match:
                drive_letter = path_val[0].lower()
                remainder = match.group(1).replace("\\", "/")
                path_val = f"/mnt/{drive_letter}/{remainder}"

        path_val = path_val.replace("\\", "/")
        return path_val

    def _post_process_settings(self):
        if "train_models" in self._settings:
            self._settings["dpipe_model_dict"] = {model: model for model in self._settings["train_models"]}

        if (
            "THUMB_TARGET_W" in self._settings
            and "THUMB_TARGET_H" in self._settings
            and self._settings["THUMB_TARGET_H"] != 0
        ):
            self._settings["TARGET_ASPECT_RATIO"] = (
                self._settings["THUMB_TARGET_W"] / self._settings["THUMB_TARGET_H"]
            )
        else:
            self._settings["TARGET_ASPECT_RATIO"] = 16 / 9

        if "VIDEO_EXTENSIONS" in self._settings and "IMAGE_EXTENSIONS" in self._settings:
            self._settings["MEDIA_EXTENSIONS"] = (
                self._settings["VIDEO_EXTENSIONS"] + self._settings["IMAGE_EXTENSIONS"]
            )
        else:
            self._settings["MEDIA_EXTENSIONS"] = []

        # Determine project root. Prefer explicit 'project_location' from settings.json
        default_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        configured = self._settings.get("project_location")
        if isinstance(configured, str) and configured.strip():
            project_root = os.path.abspath(self._normalize_path_for_platform(configured.strip()))
        else:
            project_root = default_project_root
        # Expose project root so other modules can reference it deterministically
        self._settings["PROJECT_ROOT"] = project_root

        paths_to_absolutize = [
            "DATASETS_DIR",
            "DATASETS_IMG_DIR",
            "THUMBNAILS_BASE_DIR",
            "LORA_MODELS_DIR",
            "THUMBNAILS_IMG_BASE_DIR",
        ]

        for key in paths_to_absolutize:
            if key in self._settings and isinstance(self._settings[key], str):
                normalized_value = self._normalize_path_for_platform(self._settings[key])
                if os.path.isabs(normalized_value):
                    resolved_path = normalized_value
                else:
                    resolved_path = os.path.join(project_root, *normalized_value.split("/"))
                self._settings[key] = os.path.normpath(resolved_path)

        if "FFMPEG_PATH" in self._settings and isinstance(self._settings["FFMPEG_PATH"], str):
            ffmpeg_path_val_raw = str(self._settings["FFMPEG_PATH"])

            if ffmpeg_path_val_raw.lower() not in ["ffmpeg", "ffmpeg.exe"]:
                normalized_value = self._normalize_path_for_platform(ffmpeg_path_val_raw)
                if os.path.isabs(normalized_value):
                    resolved_path = normalized_value
                else:
                    resolved_path = os.path.join(project_root, *normalized_value.split("/"))
                self._settings["FFMPEG_PATH"] = os.path.normpath(resolved_path)
            else:
                self._settings["FFMPEG_PATH"] = ffmpeg_path_val_raw

    def get(self, key, default=None):
        return self._settings.get(key, default)

    def __getattr__(self, name):
        if name in self._settings:
            return self._settings[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


settings = Config()
