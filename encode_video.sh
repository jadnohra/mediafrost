ffmpeg -i "$1" -map_metadata 0 -vf scale=1280:-1 -c:v libx264 -preset slow -crf 25 -acodec libmp3lame -ab 64k "$2"
exiftool -tagsFromFile "$1" -extractEmbedded -all:all -FileModifyDate -overwrite_original "$2"
