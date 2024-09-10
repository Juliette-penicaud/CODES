


Dis_ADCP = [2352 ,2970 ,1955 , 2043  , 1841 , 1388 ,418 , 595 , 420 , 310  , 148 , 208 ]
Dis_TT = [2258,2500 ,2430, 1694  , 1602  ,1680  , 221   , 246   , 392  , 459   , 411   , 579]
Dis_TT_8am = [2300 ,2428 , 2489 ,1725, 1600,1677, 301 ,371  ,533   ,474   ,325  , 362  ]

for QADCP, Q_TT in zip(Dis_ADCP,Dis_TT):
    print(QADCP, Q_TT)
    Percentage = (1- QADCP/Q_TT)*100
    print(Percentage)
