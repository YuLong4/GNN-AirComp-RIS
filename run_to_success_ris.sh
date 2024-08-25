PYTHON_PATH=$(which python3)
CURRENT_DIR=$(pwd)

while true; do

    "$PYTHON_PATH" "$CURRENT_DIR/cvx_ris.py"

    if [ $? -eq 0 ]; then
        echo "over"
        break
    else
        echo "retry"
    fi
done