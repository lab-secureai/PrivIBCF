from src.privibcf_experiments import (
    export_manuscript_reported_baselines,
    prepare_all_selected,
    run_full_paper_suite,
)


def main():
    data = prepare_all_selected()
    export_manuscript_reported_baselines()
    run_full_paper_suite(data)


if __name__ == "__main__":
    main()
