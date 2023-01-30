#include "zero_server.h"
#include "random.h"
#include <boost/algorithm/string.hpp>
#include <iostream>

namespace minizero::zero {

using namespace minizero;
using namespace minizero::utils;

void ZeroLogger::createLog()
{
    std::string worker_file_name = config::zero_training_directory + "/Worker.log";
    std::string training_file_name = config::zero_training_directory + "/Training.log";
    worker_log_.open(worker_file_name.c_str(), std::ios::out | std::ios::app);
    training_log_.open(training_file_name.c_str(), std::ios::out | std::ios::app);

    for (int i = 0; i < 100; ++i) {
        worker_log_ << "=";
        training_log_ << "=";
    }
    worker_log_ << std::endl;
    training_log_ << std::endl;
}

void ZeroLogger::addLog(const std::string& log_str, std::fstream& log_file)
{
    log_file << TimeSystem::getTimeString("[Y/m/d_H:i:s.f] ") << log_str << std::endl;
    std::cerr << TimeSystem::getTimeString("[Y/m/d_H:i:s.f] ") << log_str << std::endl;
}

std::string ZeroWorkerSharedData::getSelfPlayGame()
{
    boost::lock_guard<boost::mutex> lock(mutex_);
    std::string self_play_game = "";
    if (!self_play_queue_.empty()) {
        self_play_game = self_play_queue_.front();
        self_play_queue_.pop();
    }
    return self_play_game;
}

bool ZeroWorkerSharedData::isOptimizationPahse()
{
    boost::lock_guard<boost::mutex> lock(mutex_);
    return is_optimization_phase_;
}

int ZeroWorkerSharedData::getModelIetration()
{
    boost::lock_guard<boost::mutex> lock(mutex_);
    return model_iteration_;
}

void ZeroWorkerHandler::handleReceivedMessage(const std::string& message)
{
    std::vector<std::string> args;
    boost::split(args, message, boost::is_any_of(" "), boost::token_compress_on);

    if (args[0] == "Info") {
        name_ = args[1];
        type_ = args[2];
        boost::lock_guard<boost::mutex> lock(shared_data_.worker_mutex_);
        shared_data_.logger_.addWorkerLog("[Worker Connection] " + getName() + " " + getType());
        if (type_ == "sp") {
            std::string job_command = "";
            job_command += "Job_SelfPlay ";
            job_command += config::zero_training_directory + " ";
            job_command += "nn_file_name=" + config::zero_training_directory + "/model/weight_iter_" + std::to_string(shared_data_.getModelIetration()) + ".pt";
            job_command += ":program_auto_seed=false:program_seed=" + std::to_string(utils::Random::randInt());
            job_command += ":program_quiet=true";
            write(job_command);
        } else if (type_ == "op") {
            write("Job_Optimization " + config::zero_training_directory);
        } else {
            close();
        }
        is_idle_ = true;
    } else if (args[0] == "SelfPlay") {
        if (message.find("SelfPlay", message.find("SelfPlay", 0) + 1) != std::string::npos) { return; }
        std::string game_record = message.substr(message.find(args[0]) + args[0].length() + 1);
        boost::lock_guard<boost::mutex> lock(shared_data_.mutex_);
        shared_data_.self_play_queue_.push(game_record);

        // print number of games if the queue already received many games in buffer
        if (shared_data_.self_play_queue_.size() % static_cast<int>(config::zero_num_games_per_iteration * 0.25) == 0) {
            boost::lock_guard<boost::mutex> lock(shared_data_.worker_mutex_);
            shared_data_.logger_.addWorkerLog("[SelfPlay Game Buffer] " + std::to_string(shared_data_.self_play_queue_.size()) + " games");
        }
    } else if (args[0] == "Optimization_Done") {
        boost::lock_guard<boost::mutex> lock(shared_data_.mutex_);
        shared_data_.model_iteration_ = stoi(args[1]);
        shared_data_.is_optimization_phase_ = false;
    } else {
        std::string error_message = message;
        std::replace(error_message.begin(), error_message.end(), '\r', ' ');
        std::replace(error_message.begin(), error_message.end(), '\n', ' ');
        shared_data_.logger_.addWorkerLog("[Worker Error] " + error_message);
        close();
    }
}

void ZeroWorkerHandler::close()
{
    if (isClosed()) { return; }

    boost::lock_guard<boost::mutex> lock(shared_data_.worker_mutex_);
    shared_data_.logger_.addWorkerLog("[Worker Disconnection] " + getName() + " " + getType());
    ConnectionHandler::close();
}

void ZeroServer::run()
{
    initialize();
    startAccept();
    std::cerr << TimeSystem::getTimeString("[Y/m/d_H:i:s.f] ") << "Server initialize over." << std::endl;

    for (iteration_ = config::zero_start_iteration; iteration_ <= config::zero_end_iteration; ++iteration_) {
        selfPlay();
        optimization();
    }

    close();
}

void ZeroServer::initialize()
{
    int seed = config::program_auto_seed ? static_cast<int>(time(NULL)) : config::program_seed;
    utils::Random::seed(seed);
    shared_data_.logger_.createLog();

    std::string nn_file_name = config::nn_file_name;
    nn_file_name = nn_file_name.substr(nn_file_name.find("weight_iter_") + std::string("weight_iter_").size());
    nn_file_name = nn_file_name.substr(0, nn_file_name.find("."));
    shared_data_.model_iteration_ = stoi(nn_file_name);
}

void ZeroServer::selfPlay()
{
    // setup
    std::string self_play_file_name = config::zero_training_directory + "/sgf/" + std::to_string(iteration_) + ".sgf";
    shared_data_.logger_.getSelfPlayFileStream().open(self_play_file_name.c_str(), std::ios::out);
    shared_data_.logger_.addTrainingLog("[Iteration] =====" + std::to_string(iteration_) + "=====");
    shared_data_.logger_.addTrainingLog("[SelfPlay] Start " + std::to_string(shared_data_.getModelIetration()));

    int num_collect_game = 0, game_length = 0;
    while (num_collect_game < config::zero_num_games_per_iteration) {
        broadCastSelfPlayJob();

        // read one selfplay game
        std::string self_play_game = shared_data_.getSelfPlayGame();
        if (self_play_game.empty()) {
            boost::this_thread::sleep(boost::posix_time::milliseconds(100));
            continue;
        } else if (!config::zero_server_accept_different_model_games && self_play_game.find("weight_iter_" + std::to_string(shared_data_.getModelIetration())) == std::string::npos) {
            // discard previous self-play games
            continue;
        }

        // save record
        std::string move_number = self_play_game.substr(0, self_play_game.find("(") - 1);
        std::string game_string = self_play_game.substr(self_play_game.find("("));
        shared_data_.logger_.getSelfPlayFileStream() << num_collect_game << " "
                                                     << move_number << " "
                                                     << game_string << std::endl;
        ++num_collect_game;
        game_length += stoi(move_number);

        // display progress
        if (num_collect_game % static_cast<int>(config::zero_num_games_per_iteration * 0.25) == 0) {
            shared_data_.logger_.addTrainingLog("[SelfPlay Progress] " +
                                                std::to_string(num_collect_game) + " / " +
                                                std::to_string(config::zero_num_games_per_iteration));
        }
    }

    stopJob("sp");
    shared_data_.logger_.getSelfPlayFileStream().close();
    shared_data_.logger_.addTrainingLog("[SelfPlay] Finished.");
    shared_data_.logger_.addTrainingLog("[SelfPlay Game Lengths] " + std::to_string(game_length * 1.0f / num_collect_game));
}

void ZeroServer::broadCastSelfPlayJob()
{
    boost::lock_guard<boost::mutex> lock(worker_mutex_);
    for (auto& worker : connections_) {
        if (!worker->isIdle() || worker->getType() != "sp") { continue; }
        worker->setIdle(false);
        worker->write("load_model " + config::zero_training_directory + "/model/weight_iter_" + std::to_string(shared_data_.getModelIetration()) + ".pt");
        worker->write("reset_actors");
        worker->write("start");
    }
}

void ZeroServer::optimization()
{
    shared_data_.logger_.addTrainingLog("[Optimization] Start.");

    std::string job_command = "";
    job_command += "weight_iter_" + std::to_string(shared_data_.getModelIetration()) + ".pkl";
    job_command += " " + std::to_string(std::max(1, iteration_ - config::zero_replay_buffer + 1));
    job_command += " " + std::to_string(iteration_);

    shared_data_.is_optimization_phase_ = true;
    while (shared_data_.isOptimizationPahse()) {
        boost::lock_guard<boost::mutex> lock(worker_mutex_);
        for (auto worker : connections_) {
            if (!worker->isIdle() || worker->getType() != "op") { continue; }
            worker->setIdle(false);
            worker->write(job_command);
        }
    }
    stopJob("op");

    shared_data_.logger_.addTrainingLog("[Optimization] Finished.");
}

void ZeroServer::stopJob(const std::string& job_type)
{
    boost::lock_guard<boost::mutex> lock(worker_mutex_);
    for (auto worker : connections_) {
        if (worker->getType() != job_type) { continue; }
        if (job_type == "sp") { worker->write("stop"); }
        worker->setIdle(true);
    }
}

void ZeroServer::close()
{
    boost::lock_guard<boost::mutex> lock(worker_mutex_);
    for (auto worker : connections_) { worker->write("quit"); }
    exit(0);
}

void ZeroServer::keepAlive()
{
    boost::lock_guard<boost::mutex> lock(worker_mutex_);
    for (auto worker : connections_) {
        worker->write("keep_alive");
    }
    startKeepAlive();
}

void ZeroServer::startKeepAlive()
{
    keep_alive_timer_.expires_from_now(boost::posix_time::minutes(1));
    keep_alive_timer_.async_wait(boost::bind(&ZeroServer::keepAlive, this));
}

} // namespace minizero::zero
