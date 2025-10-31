#!/bin/bash
# Start full pipeline in tmux session

SESSION_NAME="rap_full_pipeline"
SCRIPT_PATH="/home/ubuntu/RAP/run_full_pipeline_end_to_end.sh"

# Create tmux session if it doesn't exist
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_NAME" -x 240 -y 60
fi

# Run the pipeline script in tmux
tmux send-keys -t "$SESSION_NAME" "cd /home/ubuntu/RAP && $SCRIPT_PATH" C-m

echo "=========================================="
echo "Pipeline started in tmux session: $SESSION_NAME"
echo "=========================================="
echo ""
echo "To attach and watch progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach (keep running):"
echo "  Press Ctrl+B, then D"
echo ""
echo "To check status:"
echo "  tmux list-sessions"
echo ""
echo "To see logs:"
echo "  tail -f output/Cambridge/KingsCollege/full_pipeline_logs/full_pipeline_*.log"
echo ""
