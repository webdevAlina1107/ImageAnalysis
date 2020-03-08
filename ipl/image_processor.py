import argparse
import datetime
import os


def _parse_date(string: str):
    pattern = '%d/%m/%Y'
    try:
        return datetime.datetime.strptime(string, pattern).date()
    except ValueError as error:
        msg = f'Not a valid date: "{string}", reason: "{error}"'
        raise argparse.ArgumentTypeError(msg)


def _parse_file_path(path: str):
    if not os.path.isfile(path):
        message = f'Provided string is not a valid file path, "{path}"'
        raise argparse.ArgumentTypeError(message)
    return path


def database_view(arguments):
    pass


def visualize_clouds(arguments):
    pass


def visualize_occurrences(arguments):
    pass


def visualize_statistics(arguments):
    pass


def import_images(arguments):
    pass


def export_images(arguments):
    pass


def process_images(arguments):
    pass


def cmdline_arguments():
    parser = argparse.ArgumentParser(description='Utility to process images')
    parser.add_argument('--debug', action='store_true',
                        help='Toggles debug info printout')

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    # IMPORT SUBPARSER
    import_parser = subparsers.add_parser('import', help='Imports images into database')
    import_parser.add_argument('import_location', type=str, nargs='+',
                               metavar='PATH/TO/DATASET',
                               help='Location of file or folder to import,'
                                    ' can be specified multiple times')
    import_parser.add_argument('-cc', '--calculations_cache', dest='cache', action='store_true',
                               help='Enables calculations caching while importing image')

    # EXPORT SUBPARSER

    export_parser = subparsers.add_parser('export', help='Exports images out of database')
    export_parser.add_argument('export_location', type=str, metavar='PATH/TO/EXPORT/FOLDER',
                               help='Location of folder to export selected data')
    export_parser.add_argument('--start_date', dest='start', type=_parse_date, default=datetime.date.min,
                               help='Start date of a timeline', metavar='DD/MM/YYYY')
    export_parser.add_argument('--end_date', dest='end', type=_parse_date, default=datetime.date.max,
                               help='End date of a timeline', metavar='DD/MM/YYYY')
    selection_type = export_parser.add_mutually_exclusive_group(required=True)
    selection_type.add_argument('--id', dest='id', default=None, type=int,
                                help='ID of a field in a database')
    selection_type.add_argument('--all', action='store_true', dest='all',
                                help='Exports without filtering by id')

    # PROCESSING SUBPARSER

    processing_parser = subparsers.add_parser('process', help='Process an image in a database or file')
    processing_parser.add_argument('--file', dest='file', default=None, type=_parse_file_path,
                                   help='Path to image file', metavar='PATH/TO/IMAGE')
    processing_parser.add_argument('--id', dest='id', default=None, type=int,
                                   help='ID of processed image')
    processing_parser.add_argument('-cc', '--calculations_cache', dest='cache', action='store_true',
                                   help='Enables calculations caching')

    # DB VIEW SUBPARSER

    db_view_parser = subparsers.add_parser('view', help='View DB records')

    image_selection_group_ = db_view_parser.add_mutually_exclusive_group(required=True)
    image_selection_group_.add_argument('--id', nargs='+', type=int, dest='id',
                                        help="IDs of viewed fields' images")
    image_selection_group_.add_argument('--all', action='store_true', dest='all',
                                        help='Selects all records')
    db_view_parser.add_argument('--head', type=int, dest='head', default=None,
                                help='Max amount of printed records')
    db_view_parser.add_argument('--start_date', dest='start', type=_parse_date, default=datetime.date.min,
                                help='Start date of a timeline', metavar='DD/MM/YYYY')
    db_view_parser.add_argument('--end_date', dest='end', type=_parse_date, default=datetime.date.max,
                                help='End date of a timeline', metavar='DD/MM/YYYY')

    # VISUALIZATION SUBPARSERS

    visualization_parser = subparsers.add_parser('visualize', help='Visualize statistical data')
    visualization_subparsers = visualization_parser.add_subparsers(dest='diagram')
    visualization_subparsers.required = True

    # CLOUDS INFO VISUALIZATION SUBPARSER

    clouds_parser = visualization_subparsers.add_parser('clouds',
                                                        help='Histogram to visualize cloudiness '
                                                             'of images on a timeline')
    clouds_parser.add_argument('--id', required=True, dest='id', type=int,
                               help='ID of a processed field')
    clouds_parser.add_argument('--start_date', dest='start', default=datetime.date.min, type=_parse_date,
                               help='Start of analysed timeline', metavar='DD/MM/YYYY')
    clouds_parser.add_argument('--end_date', dest='end', default=datetime.date.max, type=_parse_date,
                               help='End of analysed timeline', metavar='DD/MM/YYYY')

    # VALUES OCCURRENCES VISUALIZATION SUBPARSER

    occurrences_parser = visualization_subparsers.add_parser('occurrences',
                                                             help='Histogram to visualize frequency'
                                                                  ' of values occurrences in an image')
    image_selection_group = occurrences_parser.add_mutually_exclusive_group(required=True)
    image_selection_group.add_argument('--file', dest='file', default=None, type=_parse_file_path,
                                       help='Path to image file', metavar='PATH/TO/IMAGE')
    image_selection_group.add_argument('--id', dest='id', default=None, type=int,
                                       help='ID of processed image')

    # STATISTICS VISUALIZATION SUBPARSER

    statistics_parser = visualization_subparsers.add_parser('statistics',
                                                            help='Diagram to visualize multiple'
                                                                 ' datasets statistical data')
    statistics_parser.add_argument('--id', nargs='+', required=True, type=int,
                                   help='IDs of visualized fields')
    statistics_parser.add_argument('--start_date', dest='start', default=datetime.date.min, type=_parse_date,
                                   help='Start of analysed timeline', metavar='DD/MM/YYYY')
    statistics_parser.add_argument('--end_date', dest='end', default=datetime.date.max, type=_parse_date,
                                   help='End of analysed timeline', metavar='DD/MM/YYYY')

    arguments = parser.parse_args()

    if arguments.command == 'process':
        if arguments.file and (arguments.id or arguments.cache):
            parser.error('Unable to parse mutually exclusive group ["file"] and ["id", "calculations_cache"]')

    return arguments


def main():
    cmdline_arguments()


if __name__ == '__main__':
    main()
