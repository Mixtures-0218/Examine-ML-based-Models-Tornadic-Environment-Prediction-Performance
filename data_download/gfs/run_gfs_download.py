import sys
from herbie import Herbie
from datetime import datetime, timedelta


def download_gfs(target_datetime_str, model='gfs', product='0.5-degree', save_dir='./data'):
    """
    根据目标时间，自动下载对应的GFS预报数据。
    """
    # 1. 解析目标时间
    target_dt = datetime.strptime(target_datetime_str, '%Y-%m-%d %H:%M')
    
    # 2. GFS模型每日在 00, 06, 12, 18 UTC 运行 (共4次)
    #    找到目标时间之前，最近的一次模型初始化时间
    run_hours = [0, 6, 12, 18]
    target_hour = target_dt.hour
    init_hour = max([h for h in run_hours if h <= target_hour])
    init_dt = target_dt.replace(hour=init_hour, minute=0, second=0, microsecond=0)
    
    # 3. 计算预报时效 (fxx)
    fxx = int((target_dt - init_dt).total_seconds() / 3600)
    
    print(f"目标时间: {target_dt} (UTC)")
    print(f"对应的模型初始化时间: {init_dt} (UTC)")
    print(f"预报时效 (fxx): {fxx} 小时")
    
    # 4. 使用 Herbie 准备下载
    H = Herbie(
        init_dt,
        model=model,
        product=product,
        fxx=fxx,
        save_dir=save_dir,
    )
    
    # 5. 执行下载（如果不指定变量，Herbie将下载整个GRIB2文件）
    local_filename = H.download()
    print(f"下载成功，文件保存在: {local_filename}")

if __name__ == "__main__":
    # 你可以修改这里的日期和时间
    if len(sys.argv) > 1:
        target_time = sys.argv[1]
    else:
        # 默认为你提供的示例时间
        target_time = "2015-05-07 03:00"
    
    download_gfs(target_time)