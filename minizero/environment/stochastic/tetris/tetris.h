#pragma once

#include "board.h"
#include "stochastic_env.h"
#include <algorithm>
#include <string>
#include <vector>

namespace minizero::env::tetris {

const std::string kTetrisName = "tetris";
const int kTetrisNumPlayer = 1;
const int kTetrisBoardWidth = 10;
const int kTetrisBoardHeight = 22; // 20 + 2 buffer lines
const int kTetrisDiscreteValueSize = 601;
const int kTetrisActionSize = 6; // 4 ULRD + 1 drop + 1 no_action
const int kTetrisDropActionID = kTetrisActionSize - 2;
const int kTetrisNopActionID = kTetrisActionSize - 1;
const int kTetrisChanceEventSize = 9; // 7 tetromino types + 1 fall + 1 no_action
const std::string kTetrisActionName[] = {"up", "left", "right", "down", "drop", "nop", "null"};

class TetrisAction : public BaseAction {
public:
    TetrisAction() : BaseAction() {}
    TetrisAction(int action_id, Player player) : BaseAction(action_id, player) {}
    TetrisAction(const std::vector<std::string>& action_string_args)
    {
        assert(action_string_args.size() && action_string_args[0].size());
        player_ = Player::kPlayer1;
        if (action_string_args[0] == "drop") {
            action_id_ = kTetrisDropActionID;
            return;
        }
        if (action_string_args[0] == "nop") {
            action_id_ = kTetrisNopActionID;
            return;
        }
        action_id_ = std::string("ULRD").find(std::toupper(action_string_args[0][0]));
    }

    inline Player nextPlayer() const override { return Player::kPlayer1; }
    inline std::string toConsoleString() const
    {
        return kTetrisActionName[action_id_];
    }

    int getRotation() const 
    {
        return (action_id_ == 0);
    }
    int getMovement() const
    {
        if (action_id_ == 1) return -1; // L
        if (action_id_ == 2) return 1; // R
        return 0;
    }
};

class TetrisEnv : public StochasticEnv<TetrisAction> {
public:
    TetrisEnv() {}

    void reset() override { reset(utils::Random::randInt()); }
    void reset(int seed) override;
    bool act(const TetrisAction& action, bool with_chance = true) override;
    bool act(const std::vector<std::string>& action_string_args, bool with_chance = true) override { return act(TetrisAction(action_string_args), with_chance); }
    bool actChanceEvent(const TetrisAction& action) override;
    bool actChanceEvent();

    std::vector<TetrisAction> getLegalActions() const override;
    std::vector<TetrisAction> getLegalChanceEvents() const override;
    int getMaxChanceEventSize() const override { return kTetrisChanceEventSize; } 
    float getChanceEventProbability(const TetrisAction& action) const override;
    bool isLegalAction(const TetrisAction& action) const override;
    bool isLegalChanceEvent(const TetrisAction& action) const override;
    bool isTerminal() const override;
    void lockPiece();

    int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const TetrisAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { return 2; } // feature channel nums
    inline int getNumActionFeatureChannels() const override { return 1; }
    inline int getInputChannelHeight() const override { return kTetrisBoardHeight; }
    inline int getInputChannelWidth() const override { return kTetrisBoardWidth; }
    inline int getHiddenChannelHeight() const override { return kTetrisBoardHeight; }
    inline int getHiddenChannelWidth() const override { return kTetrisBoardWidth; }
    inline int getPolicySize() const override { return kTetrisActionSize; }
    inline int getDiscreteValueSize() const override { return kTetrisDiscreteValueSize; }

    std::string toString() const override;
    std::string name() const override { return kTetrisName; }
    int getNumPlayer() const override { return kTetrisNumPlayer; }
    float getReward() const override { return reward_; }
    int getTotalLines() const { return total_lines_; }
    float getEvalScore(bool is_resign = false) const override { return total_reward_; }

private:
    struct TetrisChanceEvent {
        enum class EventType { Fall,
                               NewPiece, 
                               NoAction };
        EventType type_;
        int piece_type_;

        TetrisChanceEvent(EventType type, int piece_type = -1)
            : type_(type), piece_type_(piece_type) {}

        static TetrisChanceEvent toChanceEvent(const TetrisAction& action);
        static TetrisAction toAction(const TetrisChanceEvent& event);
    };

private:
    TetrisBoard board_;
    int reward_;
    int total_reward_;
    int total_lines_;
    void generateNewPiece(int piece_type);
    bool fall();
};

class TetrisEnvLoader : public StochasticEnvLoader<TetrisAction, TetrisEnv> {
public:
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override { return TetrisEnv().getActionFeatures(pos < static_cast<int>(action_pairs_.size()) ? action_pairs_[pos].first : TetrisAction(), rotation); }
    std::vector<float> getValue(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(calculateNStepValue(pos)) : 0.0f); }
    std::vector<float> getAfterstateValue(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(calculateNStepValue(pos) - BaseEnvLoader::getReward(pos)[0]) : 0.0f); }
    std::vector<float> getReward(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(BaseEnvLoader::getReward(pos)[0]) : 0.0f); }
    float getPriority(const int pos) const override { return fabs(calculateNStepValue(pos) - BaseEnvLoader::getValue(pos)[0]); }

    std::string name() const override { return kTetrisName; }
    int getPolicySize() const override { return kTetrisActionSize; }
    int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }

private:
    float calculateNStepValue(const int pos) const;
    std::vector<float> toDiscreteValue(float value) const;
};

} // namespace minizero::env::tetris
