# This script is used to anonymize data from the L2 Political Dataset
# Imports:
import pandas as pd
import random
import numpy as np


# Loading in sample file:
df = pd.read_csv(r'VM2--LA--2020-02-27-DEMOGRAPHIC.tab', sep='\t', lineterminator='\r')
# Do some elementary analysis of the data:
print(df.head())
print(df.columns)

# Voter ID and All telephone data can be dropped as it is unique for every individual and will not provide much
# analytical use. We can also drop voter name as this won't provide much analytical use either.
# Lastly we should drop locations that correspond to a user's street adress, location data in this format is not
# useful for analysis and is highly identifiable
to_drop = ['SEQUENCE',
           'LALVOTERID',
           'Voters_StateVoterID',
           'Voters_CountyVoterID',
           'VoterTelephones_LandlineAreaCode',
           'VoterTelephones_Landline7Digit',
           'VoterTelephones_LandlineFormatted',
           'VoterTelephones_LandlineUnformatted',
           'VoterTelephones_LandlineConfidenceCode',
           'VoterTelephones_CellPhoneOnly',
           'VoterTelephones_CellPhoneFormatted',
           'VoterTelephones_CellPhoneUnformatted',
           'VoterTelephones_CellConfidenceCode',
           'Voters_FirstName',
           'Voters_MiddleName',
           'Voters_LastName',
           'Voters_NameSuffix',
           'Residence_Addresses_AddressLine',
           'Residence_Addresses_ExtraAddressLine',
           'Residence_Addresses_City',
           'Residence_Addresses_State',
           'Residence_Addresses_Zip',
           'Residence_Addresses_ZipPlus4',
           'Residence_Addresses_HouseNumber',
           'Residence_Addresses_PrefixDirection',
           'Residence_Addresses_StreetName',
           'Residence_Addresses_Designator',
           'Residence_Addresses_SuffixDirection',
           'Residence_Addresses_ApartmentNum',
           'Residence_Addresses_ApartmentType',
           'Residence_Addresses_CassErrStatCode',
           'Voters_SequenceZigZag',
           'Voters_SequenceOddEven',
           'Residence_Addresses_CensusTract',
           'Residence_Addresses_CensusBlockGroup',
           'Residence_Addresses_CensusBlock',
           'Residence_Addresses_Property_LandSq_Footage',
           'Residence_Addresses_Property_HomeSq_Footage',
           'Residence_Addresses_Density',
           'Residence_Families_FamilyID',
           'Residence_Families_HHCount',
           'Residence_HHGender_Description',
           'Residence_HHParties_Description',
           'Mailing_Addresses_AddressLine',
           'Mailing_Addresses_ExtraAddressLine',
           'Mailing_Addresses_City',
           'Mailing_Addresses_State',
           'Mailing_Addresses_Zip',
           'Mailing_Addresses_ZipPlus4',
           'Mailing_Addresses_HouseNumber',
           'Mailing_Addresses_PrefixDirection',
           'Mailing_Addresses_StreetName',
           'Mailing_Addresses_Designator',
           'Mailing_Addresses_SuffixDirection',
           'Mailing_Addresses_ApartmentNum',
           'Mailing_Addresses_ApartmentType',
           'Mailing_Addresses_CassErrStatCode',
           'Mailing_Families_FamilyID',
           'Mailing_Families_HHCount',
           'Mailing_HHGender_Description',
           'Mailing_HHParties_Description',
           'Voters_BirthDate',
           'DateConfidence_Description',
           'VoterParties_Change_Changed_Party',
           'EthnicGroups_EthnicGroup1Desc',
           'CountyEthnic_LALEthnicCode',
           'CountyEthnic_Description',
           'Religions_Description',
           'Voters_CalculatedRegDate',
           'Voters_OfficialRegDate',
           'Voters_PlaceOfBirth',
           'Languages_Description',
           'AbsenteeTypes_Description',
           'MilitaryStatus_Description',
           'MaritalStatus_Description', ]
for i in to_drop:
    df = df.drop(i, 1)
# However, we can keep coordinates for location data so long as we hash these coordinates to make them
# unidentifiable: When hashing location data we should use LSH (location sensitive hashing) which effectively
# blurrs/obscures these coordinate:

def hash_coords(lat, long):
    Range = random(100, 1000)  # pick a random offset distance between 100 and 1000
    disrupted_x = hash(lat) % Range
    disrupted_y = hash(long) % Range
    # Randomly add/subtract range
    coords = (lat + disrupted_x * random.choice([-1, 1]), long + disrupted_y * random.choice([-1, 1]))
    return coords


for i in range(len(df['Residence_Addresses_Latitude'])):
    new_coords = hash_coords(df['Residence_Addresses_Latitude'][i], df['Residence_Addresses_Longitude'][i])
    df['Residence_Addresses_Latitude'][i], df['Residence_Addresses_Longitude'][i] = new_coords[0], new_coords[1]


# We can retain gender, to hash this data we can pick randomly between 0 and 1 and asssign one to male and the other
# to female:
def hash_gender(gender_column):
    male = random.randint(0, 1)
    for i in gender_column:
        if i == 'M':
            i = male
        else:
            i = 1 - male
    return gender_column


df['Voters_Gender'] = hash_gender(df['Voters_Gender'])


# For age we simply add a random integer between 0 and 10 to all ages:
def hash_age(age_column):
    offset = random(1, 10)
    for i in age_column:
        i += offset
    return age_column


df['Voters_Age'] = hash_age(df['Voters_Age'])


# For political party we can do the same thing that we did with gender:
def hash_party(party_column):
    repub = random.randint(0, 1)
    for i in party_column:
        if i == 'Republican':
            i = repub
        elif i == 'Democrat':
            i = 1 - repub
        else:
            i = 0.5
    return party_column


df['Parties_Description'] = hash_party(df['Parties_Description'])


# For ethnic background we can simply count the different occurances and implement a random hash using labelbinarizer:

def hash_ethnicity(ethnicity_column):
    ethnicity_column = ethnicity_column.astype('category')
    ethnicity_column = ethnicity_column.cat.codes
    return ethnicity_column


df['Ethnic_Description'] = hash_ethnicity(df['Ethnic_Description'])


# We can also maintain other districts as they are sufficiently anonymized given our coordinate hash, however just to
# be safe we can also binarize these values as well:

def hash_districts(district_column):
    district_column = district_column.astype('category')
    district_column = district_column.cat.codes
    return district_column


# iterate through districts and hash them:
district_col_names = ['US_Congressional_District',
                      'AddressDistricts_Change_Changed_CD',
                      'State_Senate_District',
                      'AddressDistricts_Change_Changed_SD',
                      'State_House_District',
                      'AddressDistricts_Change_Changed_HD',
                      'State_Legislative_District',
                      'AddressDistricts_Change_Changed_LD',
                      '2001_US_Congressional_District',
                      '2001_State_Senate_District',
                      '2001_State_House_District',
                      '2001_State_Legislative_District',
                      'County',
                      'Voters_FIPS',
                      'AddressDistricts_Change_Changed_County',
                      'Precinct',
                      'County_Commissioner_District',
                      'County_Supervisorial_District',
                      'County_Legislative_District',
                      'City',
                      'City_Council_Commissioner_District',
                      'City_Ward',
                      'City_Mayoral_District',
                      'Town_District',
                      'Town_Ward',
                      'Town_Council',
                      'Village',
                      'Village_Ward',
                      'Township',
                      'Township_Ward',
                      'Borough',
                      'Borough_Ward',
                      'Hamlet_Community_Area',
                      '4H_Livestock_District',
                      'Airport_District',
                      'Annexation_District',
                      'Aquatic_Center_District',
                      'Aquatic_District',
                      'Assessment_District',
                      'Bay_Area_Rapid_Transit',
                      'Board_of_Education_District',
                      'Board_of_Education_SubDistrict',
                      'Bonds_District',
                      'Career_Center',
                      'Cemetery_District',
                      'Central_Committee_District',
                      'Chemical_Control_District',
                      'City_School_District',
                      'Coast_Water_District',
                      'College_Board_District',
                      'Committee_Super_District',
                      'Communications_District',
                      'Community_College',
                      'Community_College_Commissioner_District',
                      'Community_College_SubDistrict',
                      'Community_College_At_Large',
                      'Community_Council_District',
                      'Community_Council_SubDistrict',
                      'Community_Facilities_District',
                      'Community_Facilities_SubDistrict',
                      'Community_Hospital_District',
                      'Community_Planning_Area',
                      'Community_Service_District',
                      'Community_Service_SubDistrict',
                      'Congressional_Township',
                      'Conservation_District',
                      'Conservation_SubDistrict',
                      'Consolidated_Water_District',
                      'Control_Zone_District',
                      'Corrections_District',
                      'County_Board_of_Education_District',
                      'County_Board_of_Education_SubDistrict',
                      'County_Community_College_District',
                      'County_Fire_District',
                      'County_Hospital_District',
                      'County_Library_District',
                      'County_Memorial_District',
                      'County_Paramedic_District',
                      'County_Service_Area',
                      'County_Service_Area_SubDistrict',
                      'County_Sewer_District',
                      'County_Superintendent_of_Schools_District',
                      'County_Unified_School_District',
                      'County_Water_District',
                      'County_Water_Landowner_District',
                      'County_Water_SubDistrict',
                      'Democratic_Convention_Member',
                      'Democratic_Zone',
                      'Designated_Market_Area_DMA',
                      'District_Attorney',
                      'Drainage_District',
                      'Education_Commission_District',
                      'Educational_Service_District',
                      'Educational_Service_Subdistrict',
                      'Election_Commissioner_District',
                      'Elementary_School_District',
                      'Elementary_School_SubDistrict',
                      'Emergency_Communication_911_District',
                      'Emergency_Communication_911_SubDistrict',
                      'Enterprise_Zone_District',
                      'Exempted_Village_School_District',
                      'EXT_District',
                      'Facilities_Improvement_District',
                      'Fire_District',
                      'Fire_Maintenance_District',
                      'Fire_Protection_District',
                      'Fire_Protection_SubDistrict',
                      'Fire_Protection_Tax_Measure_District',
                      'Fire_Service_Area_District',
                      'Fire_SubDistrict',
                      'Flood_Control_Zone',
                      'Forest_Preserve',
                      'Garbage_District',
                      'Geological_Hazard_Abatement_District',
                      'Health_District',
                      'High_School_District',
                      'High_School_SubDistrict',
                      'Hospital_District',
                      'Hospital_SubDistrict',
                      'Improvement_Landowner_District',
                      'Independent_Fire_District',
                      'Irrigation_District',
                      'Irrigation_SubDistrict',
                      'Island',
                      'Judicial_Appellate_District',
                      'Judicial_Chancery_Court',
                      'Judicial_Circuit_Court_District',
                      'Judicial_County_Board_of_Review_District',
                      'Judicial_County_Court_District',
                      'Judicial_District',
                      'Judicial_District_Court_District',
                      'Judicial_Family_Court_District',
                      'Judicial_Jury_District',
                      'Judicial_Juvenile_Court_District',
                      'Judicial_Magistrate_Division',
                      'Judicial_Sub_Circuit_District',
                      'Judicial_Superior_Court_District',
                      'Judicial_Supreme_Court_District',
                      'Justice_of_the_Peace',
                      'Land_Commission',
                      'Landscaping_And_Lighting_Assessment_Distric',
                      'Law_Enforcement_District',
                      'Learning_Community_Coordinating_Council_District',
                      'Levee_District',
                      'Levee_Reconstruction_Assesment_District',
                      'Library_District',
                      'Library_Services_District',
                      'Library_SubDistrict',
                      'Lighting_District',
                      'Local_Hospital_District',
                      'Local_Park_District',
                      'Maintenance_District',
                      'Master_Plan_District',
                      'Memorial_District',
                      'Metro_Service_District',
                      'Metro_Service_Subdistrict',
                      'Metro_Transit_District',
                      'Metropolitan_Water_District',
                      'Middle_School_District',
                      'Mosquito_Abatement_District',
                      'Mountain_Water_District',
                      'Multi_township_Assessor',
                      'Municipal_Advisory_Council_District',
                      'Municipal_Court_District',
                      'Municipal_Utility_District',
                      'Municipal_Utility_SubDistrict',
                      'Municipal_Water_District',
                      'Municipal_Water_SubDistrict',
                      'Museum_District',
                      'Northeast_Soil_and_Water_District',
                      'Open_Space_District',
                      'Open_Space_SubDistrict',
                      'Other',
                      'Paramedic_District',
                      'Park_Commissioner_District',
                      'Park_District',
                      'Park_SubDistrict',
                      'Planning_Area_District',
                      'Police_District',
                      'Port_District',
                      'Port_SubDistrict',
                      'Power_District',
                      'Proposed_City',
                      'Proposed_City_Commissioner_District',
                      'Proposed_Community_College',
                      'Proposed_District',
                      'Proposed_Elementary_School_District',
                      'Proposed_Fire_District',
                      'Proposed_Unified_School_District',
                      'Public_Airport_District',
                      'Public_Regulation_Commission',
                      'Public_Service_Commission_District',
                      'Public_Utility_District',
                      'Public_Utility_SubDistrict',
                      'Rapid_Transit_District',
                      'Rapid_Transit_SubDistrict',
                      'Reclamation_District',
                      'Recreation_District',
                      'Recreational_SubDistrict',
                      'Regional_Office_of_Education_District',
                      'Republican_Area',
                      'Republican_Convention_Member',
                      'Resort_Improvement_District',
                      'Resource_Conservation_District',
                      'River_Water_District',
                      'Road_Maintenance_District',
                      'Rural_Service_District',
                      'Sanitary_District',
                      'Sanitary_SubDistrict',
                      'School_Board_District',
                      'School_District',
                      'School_District_Vocational',
                      'School_Facilities_Improvement_District',
                      'School_Subdistrict',
                      'Service_Area_District',
                      'Sewer_District',
                      'Sewer_Maintenance_District',
                      'Sewer_SubDistrict',
                      'Snow_Removal_District',
                      'Soil_And_Water_District',
                      'Soil_And_Water_District_At_Large',
                      'Special_Reporting_District',
                      'Special_Tax_District',
                      'State_Board_of_Equalization',
                      'Storm_Water_District',
                      'Street_Lighting_District',
                      'Superintendent_of_Schools_District',
                      'Transit_District',
                      'Transit_SubDistrict',
                      'TriCity_Service_District',
                      'TV_Translator_District',
                      'Unified_School_District',
                      'Unified_School_SubDistrict',
                      'Unincorporated_District',
                      'Unincorporated_Park_District',
                      'Unprotected_Fire_District',
                      'Ute_Creek_Soil_District',
                      'Vector_Control_District',
                      'Vote_By_Mail_Area',
                      'Wastewater_District',
                      'Water_Agency',
                      'Water_Agency_SubDistrict',
                      'Water_Conservation_District',
                      'Water_Conservation_SubDistrict',
                      'Water_Control__Water_Conservation',
                      'Water_Control__Water_Conservation_SubDistrict',
                      'Water_District',
                      'Water_Public_Utility_District',
                      'Water_Public_Utility_Subdistrict',
                      'Water_Replacement_District',
                      'Water_Replacement_SubDistrict',
                      'Water_SubDistrict',
                      'Weed_District',
                      'CommercialData_BookBuyerInHome', ]
for i in district_col_names:
    df[i] = hash_districts(df[i])

# Because the "ElectionsResult fields" and the "CommercialData" fields do not contain directly sensitive information
# (so long as we propperly anonymize the other data), we do not need to anonymize these factors
# Therefore we are done anonymizing our data.
