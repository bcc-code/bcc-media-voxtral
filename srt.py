

class SRTFormatter:
    """Handles SRT subtitle formatting"""

    @staticmethod
    def format_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def create_srt_entry(index: int, start_time: float, end_time: float, text: str) -> str:
        """Create a single SRT entry"""
        start_formatted = SRTFormatter.format_time(start_time)
        end_formatted = SRTFormatter.format_time(end_time)
        return f"{index}\n{start_formatted} --> {end_formatted}\n{text}\n\n"

    @staticmethod
    def parse_time(time_str: str) -> float:
        """Parse SRT time format (HH:MM:SS,mmm) to seconds"""
        time_str = time_str.strip()
        # Handle both comma and dot as decimal separator
        time_str = time_str.replace(',', '.')

        # Split into time and milliseconds parts
        if '.' in time_str:
            time_part, ms_part = time_str.split('.')
            milliseconds = float('0.' + ms_part)
        else:
            time_part = time_str
            milliseconds = 0.0

        # Parse HH:MM:SS
        parts = time_part.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])

        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds
        return total_seconds
