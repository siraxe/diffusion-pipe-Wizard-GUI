import os
import sys
import subprocess
import toml
from pathlib import Path
from typing import Dict, List, Optional, Literal
from loguru import logger

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))


class MusubiRun:
    # Model type detection patterns
    MODEL_PATTERNS = {
        'ltx-video-2': ['ltx-video-2', 'ltx2', 'ltx-video'],
        'wan22': ['wan', 'wan22', 'wan-2.2', 'wan2.2', 'wan-a14b', 'wan-a14b-i2v'],
        'minimaxh3': ['minimaxh3', 'minimax-h3', 'minimax_h3', 'mmh3'],
    }

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.config: Optional[Dict] = None
        self.config_path: Optional[str] = None
        self.model_type: Optional[str] = None
        self.cache_handler = None
        self.run_handler = None

    @staticmethod
    def _find_project_root() -> Path:
        """Auto-detect the project root directory."""
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "flet_app").exists() or (parent / "diffusion-trainers").exists():
                return parent
        return Path.cwd()

    # ==========================================================================
    # Config Loading
    # ==========================================================================

    def load_config(self, config_path: str) -> 'MusubiRun':
        self.config_path = config_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = toml.load(f)

        # Detect model type
        self.model_type = self._detect_model_type()

        # Load appropriate handlers
        self._load_handlers()

        logger.info(f"Loaded config: {config_path}")
        logger.info(f"Detected model type: {self.model_type}")

        return self

    def _detect_model_type(self) -> str:
        if not self.config:
            raise ValueError("No config loaded")

        # First, check if trainer is 'musubi' - required for musubi workflow
        trainer = self.config.get('model', {}).get('trainer', 'diffusion-pipe')
        if trainer != 'musubi':
            raise ValueError(f"Musubi workflow requires trainer='musubi', got '{trainer}'")

        # Then check model type
        model_type = self.config.get('model', {}).get('type', '').lower()

        for model_key, patterns in self.MODEL_PATTERNS.items():
            if any(pattern in model_type for pattern in patterns):
                return model_key

        # Fallback: check checkpoint path
        checkpoint = self.config.get('model', {}).get('model_path', '').lower()
        for model_key, patterns in self.MODEL_PATTERNS.items():
            if any(pattern in checkpoint for pattern in patterns):
                return model_key

        raise ValueError(f"Unable to detect supported model type. Model type: {model_type}, Trainer: {trainer}")

    def _load_handlers(self):
        try:
            if self.model_type == 'ltx-video-2':
                from musubi_utils.ltx2_cache import LTX2Cache
                from musubi_utils.ltx2_run import LTX2Run
                self.cache_handler = LTX2Cache(str(self.project_root))
                self.run_handler = LTX2Run(str(self.project_root))
            elif self.model_type == 'wan22':
                from musubi_utils.wan22_cache import WAN22Cache
                from musubi_utils.wan22_run import WAN22Run
                self.cache_handler = WAN22Cache(str(self.project_root))
                self.run_handler = WAN22Run(str(self.project_root))
            elif self.model_type == 'minimaxh3':
                from musubi_utils.mmh3_cache import MMH3Cache
                from musubi_utils.mmh3_run import MMH3Run
                self.cache_handler = MMH3Cache(str(self.project_root))
                self.run_handler = MMH3Run(str(self.project_root))
            else:
                raise ValueError(f"No handler available for model type: {self.model_type}")
        except ImportError as e:
            raise ImportError(f"Failed to import handlers for {self.model_type}: {e}")

    # ==========================================================================
    # Cache Commands
    # ==========================================================================

    def get_cache_commands(
        self,
        dataset_config: str,
        slider_config: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, List[str]]:
        if not self.cache_handler:
            raise ValueError("No cache handler loaded")

        return self.cache_handler.build_all_cache_commands(
            self.config,
            dataset_config,
            slider_config,
            output_dir or self.config.get('model', {}).get('output_dir')
        )

    def format_cache_commands(
        self,
        dataset_config: str,
        slider_config: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        if not self.cache_handler:
            raise ValueError("No cache handler loaded")

        return self.cache_handler.format_all_cache_commands(
            self.config,
            dataset_config,
            slider_config,
            output_dir or self.config.get('model', {}).get('output_dir')
        )

    # ==========================================================================
    # Training Commands
    # ==========================================================================

    def get_training_command(
        self,
        dataset_config: str,
        slider_config: Optional[str] = None,
        resume: Optional[str] = None,
        reset_optimizer: bool = False,
        reset_optimizer_params: bool = False
    ) -> List[str]:
        if not self.run_handler:
            raise NotImplementedError(f"Training handler not implemented for {self.model_type} yet")

        return self.run_handler.build_training_command(
            self.config,
            dataset_config,
            slider_config,
            resume,
            reset_optimizer,
            reset_optimizer_params
        )

    def format_training_command(
        self,
        dataset_config: str,
        slider_config: Optional[str] = None,
        resume: Optional[str] = None,
        reset_optimizer: bool = False,
        reset_optimizer_params: bool = False
    ) -> str:
        if not self.run_handler:
            raise NotImplementedError(f"Training handler not implemented for {self.model_type} yet")

        return self.run_handler.format_training_command(
            config=self.config,
            dataset_config=dataset_config,
            slider_config=slider_config,
            resume=resume,
            reset_optimizer=reset_optimizer,
            reset_optimizer_params=reset_optimizer_params
        )

    # ==========================================================================
    # Execution Methods
    # ==========================================================================

    def run_cache(
        self,
        dataset_config: str,
        mode: Literal['latents', 'text_encoder', 'sample_prompts', 'all'] = 'all',
        slider_config: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> subprocess.Popen:
        commands = self.get_cache_commands(dataset_config, slider_config, output_dir)

        if mode == 'all':
            # Run all in sequence
            return self._run_cache_sequence(commands, dataset_config)
        else:
            # Run specific cache
            if mode not in commands:
                raise ValueError(f"Unknown cache mode: {mode}")
            return self._run_command(commands[mode])

    def run_training(
        self,
        dataset_config: str,
        slider_config: Optional[str] = None,
        resume: Optional[str] = None,
        reset_optimizer: bool = False,
        reset_optimizer_params: bool = False
    ) -> subprocess.Popen:
        cmd = self.get_training_command(dataset_config, slider_config, resume, reset_optimizer, reset_optimizer_params)
        return self._run_command(cmd)

    def _run_cache_sequence(self, commands: Dict[str, List[str]], dataset_config: str) -> None:
        # Order matters
        cache_order = ['latents', 'text_encoder']
        if 'sample_prompts' in commands:
            cache_order.append('sample_prompts')

        for cache_type in cache_order:
            if cache_type in commands:
                logger.info(f"Running {cache_type} caching...")
                proc = self._run_command(commands[cache_type])
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"{cache_type} caching failed with code {proc.returncode}")

    def run_cache_async(
        self,
        dataset_config: str,
        cache_type: str = 'latents',
        slider_config: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> subprocess.Popen:
        commands = self.get_cache_commands(dataset_config, slider_config, output_dir)

        if cache_type not in commands:
            available = list(commands.keys())
            raise ValueError(f"Cache type '{cache_type}' not available. Available: {available}")

        cmd = commands[cache_type]

        # Handle list of lists (i2v_preprocess can have multiple commands for multiple video dirs)
        # For now, run the first command only
        if cmd and isinstance(cmd[0], list):
            cmd = cmd[0]
            logger.info(f"Running first of multiple {cache_type} commands")

        return self._run_command(cmd)

    def _run_command(self, cmd: List[str]) -> subprocess.Popen:
        logger.info(f"Running command: {' '.join(cmd[:3])}...")

        # Force unbuffered output for real-time streaming
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.project_root),
            env=env,
        )

        return proc

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def get_model_type(self) -> str:
        return self.model_type

    def get_config(self) -> Dict:
        """Get the loaded configuration dictionary."""
        return self.config

    def print_commands(self, dataset_config: str, slider_config: Optional[str] = None, output_dir: Optional[str] = None):
        print(f"\n=== Musubi Runner - Model: {self.model_type} ===\n")

        print("Cache Commands:")
        cache_cmds = self.format_cache_commands(dataset_config, slider_config, output_dir)
        for cmd_type, cmd_str in cache_cmds.items():
            print(f"\n[{cmd_type.upper()}]")
            print(cmd_str)

        print("\n" + "="*60)
        print("\nTraining Command:")
        train_cmd = self.format_training_command(dataset_config, slider_config)
        print(train_cmd)
        print()


# ==========================================================================
# Convenience Functions
# ==========================================================================

def create_runner(config_path: str, project_root: Optional[str] = None) -> MusubiRun:
    runner = MusubiRun(project_root)
    runner.load_config(config_path)
    return runner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Musubi Run - Universal Musubi-Trainer Wrapper")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--dataset", required=True, help="Path to dataset config file")
    parser.add_argument("--mode", choices=['cache', 'train', 'both', 'print'], default='print',
                       help="What to do (cache/train/both/print)")
    parser.add_argument("--cache-type", choices=['latents', 'text_encoder', 'sample_prompts', 'all'],
                       default='all', help="Which cache to run")
    parser.add_argument("--resume", help="Path to state directory for resuming")

    args = parser.parse_args()

    runner = create_runner(args.config)

    if args.mode == 'print':
        runner.print_commands(args.dataset)
    elif args.mode == 'cache':
        runner.run_cache(args.dataset, mode=args.cache_type)
    elif args.mode == 'train':
        runner.run_training(args.dataset, resume=args.resume)
    elif args.mode == 'both':
        runner.run_cache(args.dataset)
        runner.run_training(args.dataset)
