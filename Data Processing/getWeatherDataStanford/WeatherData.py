# this file uses public sources to determine time series weather data for buildings. This 
# is pieced together using:
# 1) The GreenButton data uploaded by the user. This will define the time range for the weather 
#    data needs and is likely, but not guaranteed to contain zip code and even address data.
# 2) the lat/lon location of a building based on its ZipCode: 
#    see http://jeffreybreen.wordpress.com/2010/12/11/geocode-zip-codes/
# 3) the best match(es) for weather stations by using proximity via lat/lon to WBAN lists: 
#    see ftp://ftp.ncdc.noaa.gov/pub/data/inventories/WBAN.TXT or 
#    the stationsYEARMO.txt files in these http://cdo.ncdc.noaa.gov/qclcd_ascii/
# 4) daily (and optionally hourly and monthly) weather summaries for the appropriate time range
#    also from these: http://cdo.ncdc.noaa.gov/qclcd_ascii/
#    Note that the wclcd files are monthly and therefore requests will span several.
import csv
import os
import urllib
import zipfile
import numpy as np
import datetime
import math

class WeatherData(object):
    def __init__(self, dataDir):
        self.ZIP_MAP = None  # Lazy initialization of zip map later
        self.DATA_DIR = dataDir
        self.WBAN_FILE = os.path.join(dataDir, 'WBAN.TXT')
        self.ZIP5_FILE = os.path.join(dataDir, 'Erle_zipcodes.csv')
        self.GAZ_ZCTA_FILE = os.path.join(dataDir, '2015_Gaz_zcta_national.zip')
        self.GAZ_INNER_FILE = '2015_Gaz_zcta_national.txt'
        self.ZCDB_2012_FILE = os.path.join(dataDir, 'free-zipcode-database-Primary.zip')
        self.ZCDB_2012_INNER_FILE = 'free-zipcode-database-Primary.csv'
        self.NOAA_QCLCD_DATA_DIR = 'http://www.ncdc.noaa.gov/orders/qclcd/'

    def zipMap(self):
        if self.ZIP_MAP is None:
            zipList = self.csvData(self.ZIP5_FILE, skip=1)
            self.ZIP_MAP = {}
            for zipRow in zipList:
                self.ZIP_MAP[int(zipRow[0])] = (float(zipRow[3]), float(zipRow[4]))

            gazZCTA = self.zippedData(self.GAZ_ZCTA_FILE, self.GAZ_INNER_FILE, delim='\t', skip=1)
            for gazRow in gazZCTA:
                self.ZIP_MAP.setdefault(int(gazRow[0].strip()), (float(gazRow[5].strip()), float(gazRow[6].strip())))

            zcdb = self.zippedData(self.ZCDB_2012_FILE, self.ZCDB_2012_INNER_FILE, delim=',', skip=1)
            for zc in zcdb:
                try:
                    self.ZIP_MAP.setdefault(int(zc[0].strip()), (float(zc[5].strip()), float(zc[6].strip())))
                except ValueError:
                    pass
            print(f'Zip to lat/long lookup initialized with {len(self.ZIP_MAP)} entries')
        return self.ZIP_MAP

    def weatherUrl(self, year, month):
        return f'{self.NOAA_QCLCD_DATA_DIR}QCLCD{year}{month:02d}.zip'

    def weatherZip(self, year, month):
        return os.path.join(self.DATA_DIR, f'QCLCD{year}{month:02d}.zip')

    def hourlyFile(self, year, month):
        return f'{year}{month:02d}hourly.txt'

    def dailyFile(self, year, month):
        return f'{year}{month:02ddaily.txt}'

    def stationFile(self, year, month):
        return f'{year}{month:02d}station.txt'

    def summarizeStation(self, stationData):
        return f"{stationData[0]}, {stationData[1]:06.2f}, {stationData[9]}"

    def confirmedWeatherZip(self, year, month):
        retrieveFile = False
        filePath = self.weatherZip(year, month)
        if os.path.isfile(filePath):
            now = datetime.datetime.now()
            postYear = year
            postMonth = (month + 1) % 13
            if postMonth == 0:
                postYear += 1
                postMonth = 1
            postDate = datetime.datetime(postYear, postMonth, 7)
            modTime = datetime.datetime.fromtimestamp(os.path.getmtime(filePath))
            if postDate >= modTime:
                if now.date() != modTime.date():
                    retrieveFile = True
        else:
            retrieveFile = True

        if retrieveFile:
            url = self.weatherUrl(year, month)
            print(f'{filePath} not found. Attempting download at {url}')
            urllib.request.urlretrieve(url, filePath)

        return filePath

    def csvData(self, filePath, delim=',', colVal=None, subset=None, skip=0):
        with open(filePath, 'r', encoding='utf-8') as f:
            return self.csvDump(f, delim, colVal, subset, skip)

    def csvDump(self, f, delim=',', colVal=None, subset=None, skip=0):
        fReader = csv.reader(f, delimiter=delim)
        for _ in range(skip):
            next(fReader)  # Skip the specified number of lines
        out = []
        if colVal:
            filterColIdx = colVal[0]
            filterValues = list(colVal[1])

            if subset is None:
                out = [row for row in fReader if len(row) > 0 and row[filterColIdx] in filterValues]
            else:
                out = [[row[i] for i in subset] for row in fReader if len(row) > 0 and row[filterColIdx] in filterValues]
        else:
            out = [row for row in fReader if len(row) > 0]
        return out

    def zippedData(self, filePath, innerFile, delim=',', colVal=None, subset=None, skip=0):
        with zipfile.ZipFile(filePath, 'r') as zf:
            return self.csvDump(zf.open(innerFile), delim, colVal, subset, skip)

    def stationData(self, y, m, colVal=None, subset=None, skip=0):
        return self.zippedData(self.confirmedWeatherZip(y, m), self.stationFile(y, m), '|', colVal, subset, skip)

    def dailyData(self, y, m, colVal=None, subset=None, skip=0):
        return self.zippedData(self.confirmedWeatherZip(y, m), self.dailyFile(y, m), ',', colVal, subset, skip)

    def hourlyData(self, y, m, colVal=None, subset=None, skip=0):
        return self.zippedData(self.confirmedWeatherZip(y, m), self.hourlyFile(y, m), ',', colVal, subset, skip)

    def distLatLon(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))  # solid angle between points
        km = 6367 * c
        return km 
  
  # find the WBAN of the weather station closest to the zip code in question
  # using the zip5 lat/lon and the station lat/lon
  # this is potentially diferent every month. Bummer.
class WeatherData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Initialize any other data or attributes needed for WeatherData class

    def zip_map(self):
        # Example method to return a map of zip codes
        # Assuming it fetches data for zip code mapping from a source
        return {12601: "Poughkeepsie, NY", 90210: "Beverly Hills, CA"}  # Example mapping
    
    def match_dates(self, dates, wDates):
        """Find the indices for each list where they share the same values."""
        j = 0  # Index of wDates
        wIdx = []  # List to store indices for wDates
        dIdx = []  # List to store indices for dates
        
        try:
            for i, d in enumerate(dates):
                while j < len(wDates) and wDates[j] < d:
                    j += 1
                if j < len(wDates) and wDates[j] == d:
                    wIdx.append(j)
                    dIdx.append(i)
        except IndexError:
            pass  # If we run out of wDates, the loop should end
        
        return dIdx, wIdx  # Return the matching indices

    def float_parse(self, string, fail=np.nan):
        """Attempts to convert a string to float, returns fail if it fails."""
        try:
            return float(string)
        except ValueError:
            return fail

    def weather_range(self, zip_code, start, end, hourly=False):
        """Example method to fetch weather data between start and end dates."""
        # For simplicity, we'll generate dummy weather data.
        # Replace this with actual logic to fetch data from a source (API or file).
        
        dates = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]
        weather_data = []
        
        for date in dates:
            weather_data.append({
                'date': date,
                'temperature': np.random.uniform(30, 90),  # Dummy temperature in F
                'humidity': np.random.uniform(40, 80),     # Dummy humidity percentage
            })
        
        return weather_data

    def daily_data(self, year, month, col_val=None, subset=None):
        """Example method to fetch daily weather data for a given year and month."""
        # Generate dummy data, replace this with actual fetching logic.
        start = datetime.datetime(year, month, 1)
        end = datetime.datetime(year, month, 1) + datetime.timedelta(days=30)
        return self.weather_range(None, start, end)  # Fetch data using weather_range

class Timer:
    """Context manager for measuring elapsed time."""
    
    def __init__(self, label):
        self.label = label
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = datetime.datetime.now() - self.start_time
        print(f"{self.label} took {elapsed_time.total_seconds():.2f} seconds")

if __name__ == '__main__':
    # Create a WeatherData object with a dummy data directory
    wd = WeatherData('weather')
    
    zip5 = 12601
    zips = wd.zip_map()  # Map zip codes to their locations
    print(f"Weather data for zip code {zip5}: {zips.get(zip5, 'Unknown zip code')}")
    
    # Measure the time taken to fetch weather data for a date range
    with Timer('weather data fetch'):
        start = datetime.datetime(2013, 3, 1)
        end = datetime.datetime(2013, 4, 21)
        weather = wd.weather_range(zip5, start, end, hourly=True)
        print("Fetched weather data:", weather)

        # You can further process the weather data here, for example:
        for record in weather:
            print(f"Date: {record['date']}, Temperature: {record['temperature']}°F, Humidity: {record['humidity']}%")
    
    # Fetch and filter data for a specific WBAN (weather station)
    with Timer('filtered weather data'):
        weather_wban = wd.daily_data(2013, 3, col_val=(0, '03013'))
        print(f"Weather data for WBAN 03013: {weather_wban}")
    
    # Example of subsetting data (selecting specific columns, e.g., temperature and humidity)
    with Timer('subsetting data'):
        weather_sub = wd.daily_data(2013, 3, subset=[0, 1, 2, 4, 6])
        print(f"Subsetted weather data: {weather_sub}")