import sys, getopt

if __name__ == '__main__':
  cfgFile  = None
  outFile  = None
  n        = 3
  prefDist = 30
  query    = False
  queryZip = None
  instruction = '''Usage:
python %s -i <inputfile> -o <outputfile> -n <stations per location> -d <preferred distance km>
  OR
python %s -q <zipcode> -n <stations per location> -d <preferred distance km>''' % tuple([sys.argv[0]]*2)
  try:
    opts, args = getopt.getopt(sys.argv[1:],"hq:n:d:i:o:",["inputfile=","outputfile=","distance="])
  except getopt.GetoptError:             print instruction; sys.exit(2)
  for opt,arg in opts:
    if   opt == '-h':                    print instruction; sys.exit()
    elif opt in ("-i", "--inputfile" ):  cfgFile  = arg
    elif opt in ("-o", "--outputfile"):  outFile  = arg
    elif opt in ("-d", "--distance"):    prefDist = int(arg)
    elif opt == '-n':                    n        = int(arg)
    elif opt == '-q':                    query = True; queryZip = arg

  if not query and (cfgFile is None or outFile is None): print instruction; sys.exit()

  import WeatherData as weather
  import pandas as pd
  import datetime
  from dateutil import rrule
  import csv

  print '''  
Config file "%s" will be processed to file "%s".
Using %d stations per location and a preferred distance of %d km.''' % (cfgFile,outFile,n,prefDist)


  startTime = datetime.datetime.now()
  wd = weather.WeatherData('weather')

  if query:
    sList = wd.stationList(queryZip,2013,3,n=n,preferredDistKm=prefDist)
    for stationData in sList: print stationData
    sys.exit()

  # the goal of this code is to get a list of zip codes to save weather data for
  # for every month in the range of all dates that need data. To do this, we 
  # building a dict whose keys are complete set of all months as datetimes
  # and whose values are lists of zip codes.

  # step 1: build a list of zips and their start and end dates, with all dates
  # normalized to the beginning of the month.
  with open(cfgFile,'rb') as configFile:
    configData = csv.reader(configFile)
    configData.next() # bypass the headers
    fmts = ('%m/%d/%Y','%m-%d-%Y','%Y/%m/%d','%Y-%m-%d')
    dateRange = []
    monthZips = {}
    minStart = None
    maxEnd   = None
    for (zip5,startStr,endStr) in configData:
      if zip5 == '': continue
      startDt = None
      endDt = None
      for fmt in fmts:
        try:
          startDt = datetime.datetime.strptime(startStr,fmt)
        except: pass
        if startDt is not None: break
      for fmt in fmts:
        try:
          endDt = datetime.datetime.strptime(endStr,fmt)
        except: pass
        if endDt is not None: break
      start = datetime.datetime(startDt.year,startDt.month,1)
      end   = datetime.datetime(endDt.year,endDt.month,1)
      if(minStart is None or start < minStart): minStart = start
      if(maxEnd   is None or end   > maxEnd  ): maxEnd   = end
      print zip5, start, end
      #wd.stationList(zip5,2013,3,n=n,preferredDistKm=prefDist)
      dateRange.append((zip5,start,end))
  
  # step 2: create a list of months (as datetimes) spanning from the
  # earliest date to the latest date and use these as keys in a dict
  # that has the zip codes that need data for each month.
  monthDates = [x for x in rrule.rrule(rrule.MONTHLY, dtstart=minStart,until=maxEnd)]
  monthCfg = {}
  #keyMonth = datetime.datetime.strptime('2012-11-01','%Y-%m-%d')
  for mDate in monthDates:
    #if mDate != keyMonth: continue
    #print mDate
    #print dateRange[0]
    #print mDate, '>=', dateRange[0][1], mDate,'<=', dateRange[0][2]
    #print dateRange[0][1] <= mDate, dateRange[0][2] >= mDate
    zips = [dr[0] for dr in dateRange if dr[1] <= mDate and dr[2] >= mDate ]
    monthCfg[mDate] = zips
  
  # step 3: loop over every month (recall that the NOAA data is saved in monthly files)
  # and pull all data for every wban near one of the zips that need data.
  # note that the month stack is normalized in time to every hour on the hour.
  with open(outFile,'wb') as outFile:
    firstRow = True
    for key in sorted(monthCfg.keys()):
      print 'Working on month %s' % key
      zips = monthCfg[key]
      monthStack = wd.weatherMonth(zips,key.year,key.month,hourly=True,n=n,preferredDistKm=prefDist,stackData=True)
  # step 4: average all cotemperaneous observations for each zip code
  # this requires subsetting the stack of normalized observations by 
  # the wbams closest to each zip
      for zip5 in zips:
        zipStations = [x[0] for x in wd.stationList(zip5,key.year,key.month,n=n,preferredDistKm=prefDist)]
        flat = wd.combineStacks(monthStack,wbans=zipStations,addValues=[('zip5',zip5)])
        print '  Writing %s. %d rows.' % (zip5,len(flat))
  # step 5: write 1 month's worth of weather data for every zip code that 
  # needs data month by month...
        # print the headers, but only on the first row
        outFile.write(flat.to_csv(header=firstRow))
        if firstRow: firstRow = False

      print 'Elapsed time: ', datetime.datetime.now() - startTime
  # step 6: import the csv data into a database for querying...