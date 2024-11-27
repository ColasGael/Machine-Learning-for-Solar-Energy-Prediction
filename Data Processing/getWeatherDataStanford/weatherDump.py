import sys
import getopt
import csv
import datetime
from dateutil import rrule
import WeatherData as weather

if __name__ == '__main__':
    cfgFile = None
    outFile = None
    n = 3
    prefDist = 30
    query = False
    queryZip = None
    
    instruction = '''Usage:
    python %s -i <inputfile> -o <outputfile> -n <stations per location> -d <preferred distance km>
    OR
    python %s -q <zipcode> -n <stations per location> -d <preferred distance km>''' % tuple([sys.argv[0]]*2)
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hq:n:d:i:o:", ["inputfile=", "outputfile=", "distance="])
    except getopt.GetoptError:
        print(instruction)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print(instruction)
            sys.exit()
        elif opt in ("-i", "--inputfile"):
            cfgFile = arg
        elif opt in ("-o", "--outputfile"):
            outFile = arg
        elif opt in ("-d", "--distance"):
            prefDist = int(arg)
        elif opt == '-n':
            n = int(arg)
        elif opt == '-q':
            query = True
            queryZip = arg

    if not query and (cfgFile is None or outFile is None):
        print(instruction)
        sys.exit()

    print(f'''
    Config file "{cfgFile}" will be processed to file "{outFile}".
    Using {n} stations per location and a preferred distance of {prefDist} km.
    ''')

    startTime = datetime.datetime.now()
    wd = weather.WeatherData('weather')

    if query:
        sList = wd.stationList(queryZip, 2013, 3, n=n, preferredDistKm=prefDist)
        for stationData in sList:
            print(stationData)
        sys.exit()

    # Step 1: Build a list of zips and their start and end dates
    with open(cfgFile, 'r', encoding='utf-8') as configFile:
        configData = csv.reader(configFile)
        next(configData)  # Skip the headers
        fmts = ('%m/%d/%Y', '%m-%d-%Y', '%Y/%m/%d', '%Y-%m-%d')
        dateRange = []
        minStart = None
        maxEnd = None
        for zip5, startStr, endStr in configData:
            if zip5 == '':
                continue
            startDt = None
            endDt = None
            for fmt in fmts:
                try:
                    startDt = datetime.datetime.strptime(startStr, fmt)
                    break
                except ValueError:
                    pass
            for fmt in fmts:
                try:
                    endDt = datetime.datetime.strptime(endStr, fmt)
                    break
                except ValueError:
                    pass
            start = datetime.datetime(startDt.year, startDt.month, 1)
            end = datetime.datetime(endDt.year, endDt.month, 1)
            if minStart is None or start < minStart:
                minStart = start
            if maxEnd is None or end > maxEnd:
                maxEnd = end
            print(zip5, start, end)
            dateRange.append((zip5, start, end))

    # Step 2: Create a list of months spanning from the earliest to the latest date
    monthDates = [x for x in rrule.rrule(rrule.MONTHLY, dtstart=minStart, until=maxEnd)]
    monthCfg = {}
    
    for mDate in monthDates:
        zips = [dr[0] for dr in dateRange if dr[1] <= mDate and dr[2] >= mDate]
        monthCfg[mDate] = zips

    # Step 3: Loop over every month to pull all data for nearby stations
    with open(outFile, 'w', encoding='utf-8') as outFile:
        firstRow = True
        for key in sorted(monthCfg.keys()):
            print(f'Working on month {key}')
            zips = monthCfg[key]
            monthStack = wd.weatherMonth(zips, key.year, key.month, hourly=True, n=n, preferredDistKm=prefDist, stackData=True)

            # Step 4: Average all contemporaneous observations for each zip code
            for zip5 in zips:
                zipStations = [x[0] for x in wd.stationList(zip5, key.year, key.month, n=n, preferredDistKm=prefDist)]
                flat = wd.combineStacks(monthStack, wbans=zipStations, addValues=[('zip5', zip5)])
                print(f'  Writing {zip5}. {len(flat)} rows.')
                # Write headers only on the first row
                flat.to_csv(outFile, header=firstRow)
                firstRow = False

        print('Elapsed time: ', datetime.datetime.now() - startTime)
