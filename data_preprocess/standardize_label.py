import pandas as pd
import os

city_map = {'ale': ['aleppo', 'aleppo', 'sy', 'levant'], 'alg': ['algiers', 'algiers', 'dz', 'maghreb'], 'alx': ['alexandria', 'alexandria', 'eg', 'nile_basin'],
            'amm': ['amman', 'amman', 'jo', 'levant'], 'asw': ['aswan', 'aswan', 'eg', 'nile_basin'],  'alexandria': ['alexandria', 'alexandria', 'eg', 'nile_basin'],
            'assiout': ['asyut', 'asyut', 'eg', 'nile_basin'], 'assiut': ['asyut', 'asyut', 'eg', 'nile_basin'],
            'bag': ['baghdad', 'baghdad', 'iq', 'gulf'], 'bas': ['basra', 'basra', 'iq', 'gulf'], 'bei': ['beirut', 'beirut', 'lb', 'levant'], 'beirut': ['beirut', 'beirut', 'lb', 'levant'],
            'ben': ['benghazi', 'benghazi', 'ly', 'maghreb'],  'benswif': ['beni_suef', 'beni_suef', 'eg', 'nile_basin'], 'beny-swif': ['beni_suef', 'beni_suef', 'eg', 'nile_basin'], 'cai': ['cairo', 'cairo', 'eg', 'nile_basin'], 'cairo': ['cairo', 'cairo', 'eg', 'nile_basin'],
            'dam': ['damascus', 'damascus', 'sy', 'levant'], 'doh': ['doha', 'ad_dawhah', 'qa', 'gulf'], 'damanhor': ['damanhur', 'beheira', 'eg', 'nile_basin'], 'doha': ['doha', 'ad_dawhah', 'qa', 'gulf'],
            'el esmalia': ['ismailia', 'ismailia', 'eg', 'nile_basin'], 'esmalia': ['ismailia', 'ismailia', 'eg', 'nile_basin'], 'elmenia': ['minya', 'minya', 'eg', 'nile_basin'], 'elminia': ['minya', 'minya', 'eg', 'nile_basin'], 'elwasta': ['beni_suef', 'beni_suef', 'eg', 'nile_basin'],
            'fes': ['fes', 'fes_meknes', 'ma', 'maghreb'], 'guiga': ['giza', 'giza', 'eg', 'nile_basin'], 'heleopolis': ['cairo', 'cairo', 'eg', 'nile_basin'], 'helmieh': ['cairo', 'cairo', 'eg', 'nile_basin'], 'jed': ['jeddah', 'makkah', 'sa', 'gulf'], 'jer': ['jerusalem', 'west_bank', 'ps', 'levant'], 'kha': ['khartoum', 'khartoum', 'sd', 'nile_basin'],
            'luxor': ['luxor', 'luxor', 'eg', 'nile_basin'], 'mos': ['mosul', 'nineveh', 'iq', 'gulf'], 'msa': ['msa', 'msa', 'msa', 'msa'], 'mansoura': ['mansoura', 'dakahlia', 'eg', 'nile_basin'],
            'menoufia': ['shibin_el_kom', 'monufia', 'eg', 'nile_basin'], 'rab': ['rabat', 'rabat_sale_kenitra', 'ma', 'maghreb'], 'rabat': ['rabat', 'rabat_sale_kenitra', 'ma', 'maghreb'], 'sal': ['salt', 'balqa', 'jo', 'levant'], 'san': ['sanaa', 'sanaa', 'ye', 'gulf_aden'], 'sfa': ['sfax', 'sfax', 'tn', 'maghreb'], 'sfx': ['sfax', 'sfax', 'tn', 'maghreb'],
            'shartia': ['zagazig', 'sharqia', 'eg', 'nile_basin'], 'shiben_el_koom': ['shibin_el_kom', 'monufia', 'eg', 'nile_basin'], 'tanta': ['tanta', 'gharbia', 'eg', 'nile_basin'], 'tri': ['tripoli', 'north_lebanon', 'lb', 'levant'], 'tun': ['tunis', 'tunis', 'tn', 'maghreb'], 'tunis': ['tunis', 'tunis', 'tn', 'maghreb'],  'mus': ['muscat', 'muscat', 'om', 'gulf'], 'zagzig': ['zagazig', 'sharqia', 'eg', 'nile_basin'], 'nan': [], '': [], 'riy': ['riyadh', 'riyadh', 'sa', 'gulf']}
province_map = {'alg': ['algiers', 'dz', 'maghreb'], 'anb': ['annaba', 'dz', 'maghreb'], 'baghdadi': ['baghdad', 'iq', 'gulf'], 'gulf': ['gulf', 'sa', 'gulf'], 'hijazi': ['hijaz', 'sa', 'gulf'], 'msa': ['msa', 'msa', 'msa'], 'najdi': ['najd', 'sa', 'gulf'],
                'northern': ['north_iraq', 'iq', 'gulf'], 'southern': ['south_iraq', 'iq', 'gulf'], 'nan': [], 'ae_abu-dhabi': ['abu_dhabi', 'ae', 'gulf'], 'ae_dubai': ['dubai', 'ae', 'gulf'], 'ae_fujairah': ['fujairah', 'ae', 'gulf'],
                'ae_ras-al-khaymah': ['ras_al_khaimah', 'ae', 'gulf'], 'ae_umm-al-qaywayn': ['umm_al_quwain', 'ae', 'gulf'], 'bh_capital': ['capital', 'bh', 'gulf'], 'dj_djibouti': ['djibouti', 'dj', 'gulf_aden'], 'dz_bordj-bou-arreridj\u200e': ['bordj_bou_arreridj', 'dz', 'maghreb'], 'dz_bordj-bou-arreridj\\u200e': ['bordj_bou_arreridj', 'dz', 'maghreb'],
                'dz_bouira': ['bouira', 'dz', 'maghreb'], 'dz_béchar': ['bechar', 'dz', 'maghreb'], 'dz_jijel': ['jijel', 'dz', 'maghreb'],
                'dz_khenchela': ['khenchela', 'dz', 'maghreb'],
                'dz_oran': ['oran', 'dz', 'maghreb'],
                'dz_ouargla': ['ouargla', 'dz', 'maghreb'],
                'eg_alexandria': ['alexandria', 'eg', 'nile_basin'],
                'eg_aswan': ['aswan', 'eg', 'nile_basin'],
                'eg_asyut': ['asyut', 'eg', 'nile_basin'],
                'eg_beheira': ['beheira', 'eg', 'nile_basin'],
                'eg_beni-suef': ['beni_suef', 'eg', 'nile_basin'],
                'eg_cairo': ['cairo', 'eg', 'nile_basin'],
                'eg_dakahlia': ['dakahlia', 'eg', 'nile_basin'],
                'eg_faiyum': ['faiyum', 'eg', 'nile_basin'],
                'eg_gharbia': ['gharbia', 'eg', 'nile_basin'],
                'eg_ismailia': ['ismailia', 'eg', 'nile_basin'],
                'eg_kafr-el-sheikh': ['kafr_el_sheikh', 'eg', 'nile_basin'],
                'eg_luxor': ['luxor', 'eg', 'nile_basin'],
                'eg_minya': ['minya', 'eg', 'nile_basin'],
                'eg_monufia': ['monufia', 'eg', 'nile_basin'],
                'eg_north-sinai': ['north_sinai', 'eg', 'nile_basin'],
                'eg_port-said': ['port_said', 'eg', 'nile_basin'],
                'eg_qena': ['qena', 'eg', 'nile_basin'],
                'eg_red-sea': ['red_sea', 'eg', 'nile_basin'],
                'eg_sohag': ['sohag', 'eg', 'nile_basin'],
                'eg_south-sinai': ['south_sinai', 'eg', 'nile_basin'],
                'eg_suez': ['suez', 'eg', 'nile_basin'],
                'iq_al-anbar': ['al_anbar', 'iq', 'gulf'],
                'iq_al-muthannia': ['muthanna', 'iq', 'gulf'],
                'iq_an-najaf': ['najaf', 'iq', 'gulf'],
                'iq_arbil': ['erbil', 'iq', 'gulf'],
                'iq_as-sulaymaniyah': ['sulaymaniyah', 'iq', 'gulf'],
                'iq_baghdad': ['baghdad', 'iq', 'gulf'],
                'iq_basra': ['basra', 'iq', 'gulf'],
                'iq_dihok': ['duhok', 'iq', 'gulf'],
                'iq_karbala': ['karbala', 'iq', 'gulf'],
                'iq_maysan': ['maysan', 'iq', 'gulf'],
                'iq_ninawa': ['nineveh', 'iq', 'gulf'],
                'iq_wasit': ['wasit', 'iq', 'gulf'],
                'jo_aqaba': ['aqaba', 'jo', 'levant'],
                'jo_zarqa': ['zarqa', 'jo', 'levant'],
                'kw_hawalli': ['hawalli', 'kw', 'gulf'],
                'kw_jahra': ['jahra', 'kw', 'gulf'],
                'lb_akkar': ['akkar', 'lb', 'levant'],
                'lb_north-lebanon': ['north_lebanon', 'lb', 'levant'],
                'lb_south-lebanon': ['south_lebanon', 'lb', 'levant'],
                'ly_al-butnan': ['butnan', 'ly', 'maghreb'],
                'ly_al-jabal-al-akhdar': ['jabal_al_akhdar', 'ly', 'maghreb'],
                'ly_benghazi': ['benghazi', 'ly', 'maghreb'],
                'ly_misrata': ['misrata', 'ly', 'maghreb'],
                'ly_tripoli': ['tripoli', 'ly', 'maghreb'],
                'ma_marrakech-tensift-al-haouz': ['marrakesh-safi', 'ma', 'maghreb'],
                'ma_meknes-tafilalet': ['meknes_tafilalet', 'ma', 'maghreb'],
                'ma_oriental': ['oriental', 'ma', 'maghreb'],
                'ma_souss-massa-draa': ['souss_massa', 'ma', 'maghreb'],
                'ma_tanger-tetouan': ['tanger_tetouan', 'ma', 'maghreb'],
                'mr_nouakchott': ['nouakchott', 'mr', 'maghreb'],
                'msa': ['msa', 'msa', 'msa'],
                'nan': [],
                'om_ad-dakhiliyah': ['ad_dakhiliyah', 'om', 'gulf'],
                'om_al-batnah': ['al_batinah', 'om', 'gulf'],
                'om_ash-sharqiyah': ['ash_sharqiyah', 'om', 'gulf'],
                'om_dhofar': ['dhofar', 'om', 'gulf'],
                'om_musandam': ['musandam', 'om', 'gulf'],
                'om_muscat': ['muscat', 'om', 'gulf'],
                'ps_gaza-strip': ['gaza', 'ps', 'levant'],
                'ps_west-bank': ['west_bank', 'ps', 'levant'],
                'qa_ar-rayyan': ['al_rayyan', 'qa', 'gulf'],
                'qa_doha': ['doha', 'qa', 'gulf'],
                'sa_al-madinah': ['al_madinah', 'sa', 'gulf'],
                'sa_al-quassim': ['al_quassim', 'sa', 'gulf'],
                'sa_ar-riyad': ['riyadh', 'sa', 'gulf'],
                'sa_ash-sharqiyah': ['ash_sharqiyyah', 'sa', 'gulf'],
                'sa_asir': ['asir', 'sa', 'gulf'],
                'sa_hail': ['hail', 'sa', 'gulf'],
                'sa_jizan': ['jizan', 'sa', 'gulf'],
                'sa_makkah': ['makkah', 'sa', 'gulf'],
                'sa_najran': ['najran', 'sa', 'gulf'],
                'sa_tabuk': ['tabuk', 'sa', 'gulf'],
                'sd_khartoum': ['khartoum', 'sd', 'nile_basin'],
                'so_banaadir': ['banadir', 'so', 'nile_basin'],
                'sy_aleppo': ['aleppo', 'sy', 'levant'],
                'sy_as-suwayda': ['al_suwayda', 'sy', 'levant'],
                'sy_damascus-city': ['damascus', 'sy', 'levant'],
                'sy_hims': ['homs', 'sy', 'levant'],
                'sy_lattakia': ['latakia', 'sy', 'levant'],
                'tn_ariana': ['ariana', 'tn', 'maghreb'],
                'tn_kairouan': ['kairouan', 'tn', 'maghreb'],
                'tn_mahdia': ['mahdia', 'tn', 'maghreb'],
                'tn_sousse': ['sousse', 'tn', 'maghreb'],
                'ye_aden': ['aden', 'ye', 'gulf_aden'],
                'ye_al-hudaydah': ['al_hudaydah', 'ye', 'gulf_aden'],
                'ye_dhamar': ['dhamar', 'ye', 'gulf_aden'],
                'ye_ibb': ['ibb', 'ye', 'gulf_aden']}
country_map = {'ae': ['ae', 'gulf'],
               'algeria': ['dz', 'maghreb'],
               'ba': ['bh', 'gulf'],
               'bh': ['bh', 'gulf'],
               'bahrain': ['bh', 'gulf'],
               'dz': ['dz', 'maghreb'],
               'djibouti': ['dj', 'gulf_aden'],
               'eg': ['eg', 'nile_basin'],
               'egy': ['eg', 'nile_basin'],
               'egypt': ['eg', 'nile_basin'],
               'egyptian': ['eg', 'nile_basin'],
               'ga': ['gulf', 'gulf'],
               'iq': ['iq', 'gulf'],
               'irq': ['iq', 'gulf'],
               'iraq': ['iq', 'gulf'],
               'jo': ['jo', 'levant'],
               'jor': ['jo', 'levant'],
               'jordan': ['jo', 'levant'],
               'kw': ['kw', 'gulf'],
               'kuwait': ['kw', 'gulf'],
               'lb': ['lb', 'levant'],
               'leb': ['lb', 'levant'],
               'lev': ['levant', 'levant'],
               'ly': ['ly', 'levant'],
               'lebanees': ['lb', 'levant'],
               'lebanon': ['lb', 'levant'],
               'lebanon syria': ['lb, sy', 'levant'],
               'libya': ['ly', 'maghreb'],
               'ma': ['ma', 'maghreb'],
               'mar': ['ma', 'maghreb'],
               'mixed': ['mixed', 'mixed'],
               'mor': ['ma', 'maghreb'],
               'msa': ['msa', 'msa'],
               'msa (translated)': ['msa', 'msa'],
               'mauritania': ['mr', 'maghreb'],
               'morocco': ['ma', 'maghreb'],
               'morroco': ['ma', 'maghreb'],
               'om': ['om', 'gulf'],
               'oman': ['om', 'gulf'],
               'pa': ['ps', 'levant'],
               'pal': ['ps', 'levant'],
               'pl': ['ps', 'levant'],
               'palestine': ['ps', 'levant'],
               'palestinejordan': ['ps,jo', 'levant'],
               'palestinian': ['ps', 'levant'],
               'qa': ['qa', 'gulf'],
               'qatar': ['qa', 'gulf'],
               'sa': ['sa', 'gulf'],
               'sd': ['sd', 'nile_basin'],
               'sud': ['sd', 'nile_basin'],
               'sy': ['sy', 'levant'],
               'syr': ['sy', 'levant'],
               'saudi': ['sa', 'gulf'],
               'saudi arabia': ['sa', 'gulf'],
               'saudi_arabia': ['sa', 'gulf'],
               'somalia': ['so', 'nile_basin'],
               'sudan': ['sd', 'nile_basin'],
               'syria': ['sy', 'levant'],
               'tn': ['tn', 'maghreb'],
               'tun': ['tn', 'maghreb', 'maghreb'],
               'tunisia': ['tn', 'maghreb'],
               'uae': ['ae', 'gulf'],
               'united_arab_emirates': ['ae', 'gulf'],
               'ye': ['ye', 'gulf_aden'],
               'yem': ['ye', 'gulf_aden'],
               'yemen': ['ye', 'gulf_aden'],
               'jordinian': ['jo', 'levant'],
               'msa': ['msa', 'msa'],
               'nan': [],
               '': [],
               'syrian': ['sy', 'levant']}

region_map = {'eg': 'eg',
              'egy': 'eg',
              'egyptian': 'eg',
              'egyptian': 'eg',
              'ga': 'gulf',
              'gu': 'gulf',
              'gulf': 'gulf',
              'gulf': 'gulf',
              'iq': 'iraq',
              'iraqi': 'iraq',
              'lev': 'levant',
              'levantine': 'levant',
              'lv': 'levant',
              'levantine': 'levant',
              'msa': 'msa',
              'meghribi': 'maghreb',
              'north': 'levant',
              'northafrica': 'maghreb',
              'sudan': 'sd',
              'egyptian': 'eg',
              'gulf': 'gulf',
              'iraqi': 'iraq',
              'levantine': 'levant',
              'maghrebi': 'maghreb',
              'msa': 'msa',
              'nan': '', }


def standardize_labels(input_file, output_file, levels):
    """
        levels: list<string> = indicate level at which to extract standardize, possible values are "city", "province", "country", "region"
    """
    print(input_file)
    df = pd.read_csv(input_file, delimiter='\t', header=0)
    for level in levels:
        if level == 'city':
            df = standardize_city(df)
        elif level == 'province':
            df = standardize_province(df)
        elif level == 'country':
            df = standardize_country(df)
        elif level == 'region':
            df = standardize_region(df)

    print("Unique cities ", df['dialect_city_id'].unique())
    print("Unique provinces ", df['dialect_province_id'].unique())
    print("Unique countries ", df['dialect_country_id'].unique())
    print("Unique regions ", df['dialect_region_id'].unique())

    df.to_csv(output_file, index=False, sep='\t')


def standardize_region(df):
    for i in range(len(df['dialect_city_id'])):
        try:
            # Region Level
            if type(df['dialect_region_id'][i]) == str and df['dialect_region_id'][i]:
                regions = df['dialect_region_id'][i].lower().split(',')
                new_regions = []
                for r in regions:
                    new_regions.append(r.strip())
                if new_regions[0] in region_map:
                    region_label = region_map[new_regions[0]]
                    if region_label == 'eg':
                        df['dialect_country_id'][i] = 'eg'
                        df['dialect_region_id'][i] = 'nile_basin'
                        continue
                    elif region_label == 'iraq':
                        df['dialect_country_id'][i] = 'iq'
                        df['dialect_region_id'][i] = 'gulf'
                        continue
                    else:
                        df['dialect_region_id'][i] = region_label

        except:
            continue
    return df


def standardize_country(df):
    for i in range(len(df['dialect_country_id'])):
        try:
            # Country Level
            if type(df['dialect_country_id'][i]) == str and df['dialect_country_id'][i]:
                countries = df['dialect_country_id'][i].lower().split(',')
                new_countries = []
                for country in countries:
                    new_countries.append(country.strip())
                new_names = country_map[new_countries[0]]
                for j in range(len(new_countries)):
                    new_countries[j] = country_map[new_countries[j]][0]
                df['dialect_country_id'][i] = ','.join(new_countries)
                if len(new_names) == 2:
                    if (new_names[0] == 'levant'):
                        df['dialect_country_id'][i] = ''
                    df['dialect_region_id'][i] = new_names[1]
        except:
            continue
    return df


def standardize_province(df):
    for i in range(len(df['dialect_province_id'])):
        try:

            # Province Level
            if type(df['dialect_province_id'][i]) == str and df['dialect_province_id'][i]:
                provinces = df['dialect_province_id'][i].lower().split(',')
                new_provinces = []
                for p in provinces:
                    new_provinces.append(p.strip())
                new_names = province_map[new_provinces[0]]
                for j in range(len(new_provinces)):
                    new_provinces[j] = province_map[new_provinces[j]][0]
                df['dialect_province_id'][i] = ','.join(new_provinces)
                if len(new_names) == 3:
                    df['dialect_country_id'][i] = new_names[1]
                    df['dialect_region_id'][i] = new_names[2]
                    continue
        except:
            continue
    return df


def standardize_city(df):
    for i in range(len(df['dialect_city_id'])):
        try:

            # City Level
            if type(df['dialect_city_id'][i]) == str and df['dialect_city_id'][i]:
                cities = df['dialect_city_id'][i].lower().split(',')
                new_cities = []
                new_names = []
                for c in cities:
                    new_cities.append(c.strip())
                new_names = city_map[new_cities[0]]
                for j in range(len(new_cities)):
                    new_cities[j] = city_map[new_cities[j]][0]
                df['dialect_city_id'][i] = ','.join(new_cities)
                if len(new_names) == 4:
                    df['dialect_province_id'][i] = new_names[1]
                    df['dialect_country_id'][i] = new_names[2]
                    df['dialect_region_id'][i] = new_names[3]
                    continue
        except:
            continue
    return df
