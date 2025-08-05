import argparse
from pathlib import Path
import sys
from typing import Optional

from synthfin.main import SyntheticTimeSeriesPipeline
from synthfin.utils import create_config, save_config, generate_default_config
from synthfin.visualization import create_individual_plots


def main(argv: Optional[list] = None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SynthFin: Generate synthetic financial time series data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data with default configuration
  synthfin generate
  
  # Use custom configuration file
  synthfin generate -c myconfig.yaml
  
  # Quick generation with specific parameters
  synthfin generate -n 30 -d 252 --model gbm
  
  # Generate default configuration file
  synthfin init
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic time series")
    gen_parser.add_argument("-c", "--config", type=str, help="Path to configuration file")
    gen_parser.add_argument("-n", "--n-assets", type=int, help="Number of assets")
    gen_parser.add_argument("-d", "--n-days", type=int, help="Number of days")
    gen_parser.add_argument("--model", choices=["gbm", "jump_diffusion"], help="Time series model")
    gen_parser.add_argument("--correlation", choices=["naive", "hierarchical"], help="Correlation model")
    gen_parser.add_argument("-o", "--output", type=str, help="Output CSV filename")
    gen_parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    gen_parser.add_argument("--no-save", action="store_true", help="Don't save files")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Generate default configuration file")
    init_parser.add_argument("-o", "--output", type=str, default="config.yaml", 
                           help="Output filename for configuration")
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "init":
        return handle_init(args)
    elif args.command == "generate":
        return handle_generate(args)
    
    return 0


def handle_init(args):
    """
    Handle the init command.
    """
    try:
        generate_default_config(args.output)
        print(f"✓ Created default configuration file: {args.output}")
        print("\nYou can now run:")
        print(f"  synthfin generate -c {args.output}")
        return 0
    except Exception as e:
        print(f"✗ Error creating configuration: {e}", file=sys.stderr)
        return 1


def handle_generate(args):
    """
    Handle the generate command.
    """
    try:
        # Build configuration
        if args.config:
            # Load from file
            from .utils import load_config
            print(f"Loading configuration from: {args.config}")
            config = load_config(args.config)
        else:
            # Create from arguments
            config_kwargs = {}
            
            if args.n_assets:
                config_kwargs["n_assets"] = args.n_assets
            if args.n_days:
                config_kwargs["n_days"] = args.n_days
            if args.model:
                config_kwargs["time_series_model"] = args.model
            if args.correlation:
                config_kwargs["correlation_model"] = args.correlation
            if args.output:
                config_kwargs["csv_filename"] = args.output
            if args.no_viz:
                config_kwargs["enable_viz"] = False
            if args.no_save:
                config_kwargs["save_csv"] = False
                config_kwargs["save_plots"] = False
            
            config = create_config(**config_kwargs)
        
        # Run pipeline
        print("Running synthetic time series generation...")
        pipeline = SyntheticTimeSeriesPipeline()
        pipeline.config = config
        df, correlation_matrix = pipeline.run()
        
        print(f"\n✓ Successfully generated {len(df)} days of data for {len(df.columns)} assets")
        
        return 0
        
    except Exception as e:
        print(f"✗ Error generating data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())