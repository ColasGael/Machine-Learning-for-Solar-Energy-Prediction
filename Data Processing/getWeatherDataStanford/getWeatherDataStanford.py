import csv
import datetime
import WeatherData as weather

# import the weather data of the 3 closest 
if __name__ == '__main__':
    # the directory where the data will be stored
    wd = weather.WeatherData('weatherStanford')     
    # the ZipCode of Y2E2 building in Stanford
    zip5 = 94305
    # the start date of the data available from Y2E2
    start = datetime.datetime(2008,12,21)
    # the end date of the data available from Y2E2
    end   = datetime.datetime(2010,2,10)
    # we want a hourly resolution
    hourly = True
    # the columns corresponding to the weather features we want
    subset = [0,1,2]+[4,6,12,20,22,24,30,42]
    # the number of weather stations we want
    n = 3
    # the range selected for the search of closest weather station
    preferredDistKm = 15
    
    # returns the details for the n closest stations to zip5
    statList = wd.stationList(zip5,2009,01,n,preferredDistKm)
    print(statList)
    
    # store in a list the weather data wanted
    weather = wd.weatherMonths(zip5,start,end,hourly,subset,n,preferredDistKm)
    
    # store the data in csv file
    f = open("rawWeatherDataStanford.csv", "wb")
    c = csv.writer(f)

    # the list of features
    header = ["WBAN","Date","Time","SkyCondition","Visibility","Temperature","DewPoint","RelativeHumidity","WindSpeed","StationPressure","Altimeter"]
    c.writerow(header)
    c.writerows(weather)
    
    f.close()