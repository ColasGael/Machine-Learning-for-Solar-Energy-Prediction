import sys
import getopt
import csv
import datetime
from dateutil import rrule
import WeatherData as weather
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_date(date_str, formats):
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_str}")

def main(argv):
    cfg_file = None
    out_file = None
    n = 3
    pref_dist = 30
    query = False
    query_zip = None

    instruction = f"""Usage:
    python {sys.argv[0]} -i <inputfile> -o <outputfile> -n <stations per location> -d <preferred distance km>
      OR
    python {sys.argv[0]} -q <zipcode> -n <stations per location> -d <preferred distance km>"""

    try:
        opts, args = getopt.getopt(argv, "hq:n:d:i:o:", ["inputfile=", "outputfile=", "distance="])
    except getopt.GetoptError:
        print(instruction)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(instruction)
            sys.exit()
        elif opt in ("-i", "--inputfile"):
            cfg_file = arg
        elif opt in ("-o", "--outputfile"):
            out_file = arg
        elif opt in ("-d", "--distance"):
            pref_dist = int(arg)
        elif opt == "-n":
            n = int(arg)
        elif opt == "-q":
            query = True
            query_zip = arg

    if not query and (cfg_file is None or out_file is None):
        print(instruction)
        sys.exit()

    start_time = datetime.datetime.now()
    wd = weather.WeatherData("weather")

    if query:
        stations = wd.stationList(query_zip, 2013, 3, n=n, preferredDistKm=pref_dist)
        for station in stations:
            print(station)
        sys.exit()

    date_formats = ["%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d"]
    date_ranges = []
    min_start, max_end = None, None

    with open(cfg_file, "r", encoding="utf-8") as config_file:
        reader = csv.reader(config_file)
        next(reader)  # Skip headers
        for zip5, start_str, end_str in reader:
            if not zip5:
                continue
            try:
                start_date = parse_date(start_str, date_formats)
                end_date = parse_date(end_str, date_formats)
            except ValueError as e:
                logging.error(e)
                continue

            start = datetime.datetime(start_date.year, start_date.month, 1)
            end = datetime.datetime(end_date.year, end_date.month, 1)

            min_start = min(min_start or start, start)
            max_end = max(max_end or end, end)

            logging.info(f"ZIP {zip5}: {start} to {end}")
            date_ranges.append((zip5, start, end))

    month_dates = list(rrule.rrule(rrule.MONTHLY, dtstart=min_start, until=max_end))
    month_cfg = {m: [zip5 for zip5, start, end in date_ranges if start <= m <= end] for m in month_dates}

    with open(out_file, "w", encoding="utf-8") as output_file:
        first_row = True
        for month, zips in sorted(month_cfg.items()):
            logging.info(f"Processing month {month}")
            month_stack = wd.weatherMonth(
                zips, month.year, month.month, hourly=True, n=n, preferredDistKm=pref_dist, stackData=True
            )

            for zip5 in zips:
                zip_stations = [x[0] for x in wd.stationList(zip5, month.year, month.month, n=n, preferredDistKm=pref_dist)]
                flat_data = wd.combineStacks(month_stack, wbans=zip_stations, addValues=[("zip5", zip5)])
                logging.info(f"Writing data for ZIP {zip5}: {len(flat_data)} rows")
                output_file.write(flat_data.to_csv(index=False, header=first_row))
                first_row = False

    logging.info(f"Total elapsed time: {datetime.datetime.now() - start_time}")

if __name__ == "__main__":
    main(sys.argv[1:])