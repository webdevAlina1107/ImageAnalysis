import argparse
import datetime
import os
import traceback
from pprint import pformat

import ipl.importexport as importexport
from ipl._logging import configure_logger, logger
from ipl.errors import IPLError
import ipl.workflows as workflow


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


def _parse_float_in_range(string: str):
    try:
        value = float(string)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{string} not a floating-point literal")
    if 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} not in range [0.0, 1.0]")

    return value


def cmdline_arguments():
    parser = argparse.ArgumentParser(description='Utility to process images')
    parser.add_argument('--debug', action='store_true',
                        help='Toggles debug info printout')
    parser.add_argument('--log_file', dest='log_file', default=None,
                        type=str, metavar='PATH/TO/LOGS/FILE',
                        help='Path to file where to store logs')
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
    import_parser.set_defaults(function=workflow.import_images)

    # EXPORT SUBPARSER
    drivers_list = importexport.SupportedDrivers.drivers_list()

    export_parser = subparsers.add_parser('export', help='Exports images out of database')
    export_parser.add_argument('export_location', type=str, metavar='PATH/TO/EXPORT/FOLDER',
                               help='Location of folder to export selected data')
    export_parser.add_argument('--force', dest='force', action='store_true',
                               help='Create directory if not exists')
    export_parser.add_argument('--driver', type=str, choices=drivers_list,
                               dest='driver', default='GTiff',
                               help='Driver for image exporting')
    export_parser.add_argument('--start_date', dest='start', type=_parse_date, default=datetime.date.min,
                               help='Start date of a timeline', metavar='DD/MM/YYYY')
    export_parser.add_argument('--end_date', dest='end', type=_parse_date, default=datetime.date.max,
                               help='End date of a timeline', metavar='DD/MM/YYYY')
    selection_type = export_parser.add_mutually_exclusive_group(required=True)
    selection_type.add_argument('--id', dest='id_', default=None, type=int, nargs='+',
                                metavar='FIELD_ID', help='ID of a field in a database')
    selection_type.add_argument('--all', action='store_true', dest='all_',
                                help='Exports without filtering by id')
    export_parser.set_defaults(function=workflow.export_images)

    # PROCESSING SUBPARSER

    processing_parser = subparsers.add_parser('process', help='Process an image in a database or file')
    processing_parser.add_argument('--file', dest='file', default=None, type=_parse_file_path,
                                   help='Path to image file', metavar='PATH/TO/IMAGE')
    db_images_selection = processing_parser.add_mutually_exclusive_group()
    db_images_selection.add_argument('--id', dest='id_', default=None, type=int, nargs='+',
                                     metavar='IMAGE_ID', help='IDs of processed images')
    db_images_selection.add_argument('--all', dest='all_', action='store_true',
                                     help='Processes all images in database')
    processing_parser.add_argument('-cc', '--calculations_cache', dest='cache', action='store_true',
                                   help='Enables calculations caching')
    processing_parser.add_argument('--export_to', dest='export_location', default=None,
                                   type=_parse_file_path, help='Path to excel file where results would be stored')
    processing_parser.set_defaults(function=workflow.process_images)

    # DB VIEW SUBPARSER

    db_view_parser = subparsers.add_parser('view', help='View DB records')

    image_selection_group_ = db_view_parser.add_mutually_exclusive_group(required=True)
    image_selection_group_.add_argument('--id', nargs='+', type=int, dest='id_',
                                        metavar='FIELD_ID', help="IDs of viewed fields' images")
    image_selection_group_.add_argument('--all', action='store_true', dest='all_',
                                        help='Selects all records')
    db_view_parser.add_argument('--head', type=int, dest='head', default=None,
                                help='Max amount of printed records')
    db_view_parser.add_argument('--start_date', dest='start', type=_parse_date, default=datetime.date.min,
                                help='Start date of a timeline', metavar='DD/MM/YYYY')
    db_view_parser.add_argument('--end_date', dest='end', type=_parse_date, default=datetime.date.max,
                                help='End date of a timeline', metavar='DD/MM/YYYY')
    db_view_parser.set_defaults(function=workflow.database_view)

    # VISUALIZATION SUBPARSERS

    visualization_parser = subparsers.add_parser('visualize', help='Visualize statistical data')
    visualization_subparsers = visualization_parser.add_subparsers(dest='diagram')
    visualization_subparsers.required = True

    # CLOUDS INFO VISUALIZATION SUBPARSER

    clouds_parser = visualization_subparsers.add_parser('clouds',
                                                        help='Histogram to visualize cloudiness '
                                                             'of images on a timeline')
    clouds_parser.add_argument('--id', required=True, dest='id_', type=int,
                               metavar='FIELD_ID', help='ID of a processed field')
    clouds_parser.add_argument('--start_date', dest='start', default=datetime.date.min, type=_parse_date,
                               help='Start of analysed timeline', metavar='DD/MM/YYYY')
    clouds_parser.add_argument('--end_date', dest='end', default=datetime.date.max, type=_parse_date,
                               help='End of analysed timeline', metavar='DD/MM/YYYY')
    clouds_parser.set_defaults(function=workflow.visualize_clouds)

    # VALUES OCCURRENCES VISUALIZATION SUBPARSER

    occurrences_parser = visualization_subparsers.add_parser('occurrences',
                                                             help='Histogram to visualize frequency'
                                                                  ' of values occurrences in an image')
    image_selection_group = occurrences_parser.add_mutually_exclusive_group(required=True)
    image_selection_group.add_argument('--file', dest='file', default=None, type=_parse_file_path,
                                       help='Path to image file', metavar='PATH/TO/IMAGE')
    image_selection_group.add_argument('--id', dest='id_', default=None, type=int,
                                       metavar='IMAGE_ID', help='ID of processed image')
    occurrences_parser.set_defaults(function=workflow.visualize_occurrences)

    # STATISTICS VISUALIZATION SUBPARSER

    statistics_parser = visualization_subparsers.add_parser('statistics',
                                                            help='Diagram to visualize multiple'
                                                                 ' datasets statistical data')
    statistics_parser.add_argument('--id', nargs='+', required=True, type=int, dest='id_',
                                   metavar='FIELD_ID', help='IDs of visualized fields')
    statistics_parser.add_argument('--start_date', dest='start', default=datetime.date.min, type=_parse_date,
                                   help='Start of analysed timeline', metavar='DD/MM/YYYY')
    statistics_parser.add_argument('--end_date', dest='end', default=datetime.date.max, type=_parse_date,
                                   help='End of analysed timeline', metavar='DD/MM/YYYY')
    statistics_parser.add_argument('--max_cloudiness', dest='max_cloudiness', default=0.5, type=_parse_float_in_range,
                                   metavar='[0.0, 1.0]', help='Filtering cloudiness percent')
    statistics_parser.set_defaults(function=workflow.visualize_statistics)

    arguments = parser.parse_args()

    if arguments.command == 'process':
        if arguments.file and (arguments.id_ or arguments.all_) or (not arguments.file and not arguments.id_
                                                                    and not arguments.all_):
            parser.error('Unable to parse mutually exclusive group ["file"] and ["id", "calculations_cache"]')

    return arguments


def main():
    arguments = cmdline_arguments()
    configure_logger(is_debug=arguments.debug,
                     logs_file=arguments.log_file)
    try:
        function = arguments.function
        logger.debug(f'Starting target function with arguments : \n{pformat(vars(arguments), indent=4)}')
        function(**vars(arguments))
        logger.debug('Action succeeded !')
    except IPLError as error:
        logger.critical(f'Database error @ {error}')
        logger.debug(traceback.format_exc())
    except Exception as error:
        logger.critical(f'Something went wrong @ {error}')
        logger.debug(traceback.format_exc())


if __name__ == '__main__':
    main()
