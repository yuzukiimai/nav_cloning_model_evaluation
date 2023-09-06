#!/usr/bin/env python3
import pandas as pd
import glob
import os

if __name__ == '__main__':
    lists = []
    for filename in sorted(glob.glob('/home/yuzuki/40%/*/learning_exit_episode.csv'), key=lambda f: os.stat(f).st_mtime, reverse=True):
        lists.append(filename)

    combined_data = pd.DataFrame()

    for filename in lists:
        with open(filename) as f:
            data = f.read()

        values = [float(line.strip()) for line in data.split('\n') if line.strip()]

        df = pd.DataFrame({'Values': values})
        combined_data = pd.concat([combined_data, df])

    excel_writer = pd.ExcelWriter('40%.xlsx', engine='xlsxwriter')

    combined_data.to_excel(excel_writer, sheet_name='Combined Data', index=False)

    workbook = excel_writer.book
    worksheet = excel_writer.sheets['Combined Data']

    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BC',
        'border': 1
    })

    for i, column in enumerate(combined_data.columns):
        column_len = max(combined_data[column].astype(str).apply(len).max(), len(column))
        worksheet.set_column(i, i, column_len + 2)
        worksheet.write(0, i, column, header_format)

    excel_writer.save()
