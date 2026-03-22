#include <catch2/catch_all.hpp>
#include "nlp/nlp_engine.hh"
#include "nlp/addons/graph_addon.hh"
#include <memory>
#include <string>

using namespace pce::nlp;

TEST_CASE("GraphAddon: Entity Relationships and Community Detection", "[graph][leidenfold]") {
    auto model = std::make_shared<NLPModel>();
    NLPEngine engine(model);

    SECTION("Building a Knowledge Graph from Text") {
        GraphAddon graph;

        // High-context text with clear entity clusters
        // Cluster 1: Tech/Space (NASA, SpaceX, Mars)
        // Cluster 2: Biology (DNA, CRISPR, Genomics)
        std::string text =
            "NASA and SpaceX are working together on a mission to Mars. "
            "The rockets from SpaceX will carry NASA equipment to the Red Planet. "
            "Meanwhile, CRISPR technology is revolutionizing genomics. "
            "Scientists using DNA sequencing and CRISPR are mapping the human genome.";

        engine.build_knowledge_graph(text, graph, 15);
        graph.detect_communities(10);

        auto response = graph.process_impl("", {});
        REQUIRE(response.success);

        json result = json::parse(response.output);
        REQUIRE(result.is_object());
        REQUIRE(result.contains("communities"));
        REQUIRE(response.metrics.at("nodes") >= 5.0);

        bool found_tech_cluster = false;
        bool found_bio_cluster = false;

        for (auto& [id, members] : result["communities"].items()) {
            bool has_nasa = false;
            bool has_spacex = false;
            bool has_mars = false;
            bool has_crispr = false;
            bool has_dna = false;
            bool has_genomics = false;

            for (auto& member : members) {
                const std::string name = member["name"];
                if (name == "NASA") has_nasa = true;
                if (name == "SpaceX") has_spacex = true;
                if (name == "Mars") has_mars = true;
                if (name == "CRISPR") has_crispr = true;
                if (name == "DNA") has_dna = true;
                if (name == "genomics") has_genomics = true;
            }

            if (has_nasa && has_spacex && has_mars) found_tech_cluster = true;
            if (has_crispr && has_dna && has_genomics) found_bio_cluster = true;
        }

        CHECK(found_tech_cluster);
        CHECK(found_bio_cluster);
    }

    SECTION("Manual Relationship and Weighting") {
        GraphAddon manual_graph;

        // Manual construction of a "Star Wars" relationship graph
        manual_graph.add_relationship("Luke Skywalker", "Person", "Darth Vader", "Person", 5.0f);
        manual_graph.add_relationship("Luke Skywalker", "Person", "Leia Organa", "Person", 5.0f);
        manual_graph.add_relationship("Darth Vader", "Person", "Empire", "Org", 2.0f);
        manual_graph.add_relationship("Han Solo", "Person", "Chewbacca", "Person", 5.0f);
        manual_graph.add_relationship("Han Solo", "Person", "Millennium Falcon", "Ship", 3.0f);

        manual_graph.detect_communities(5);

        auto response = manual_graph.process_impl("", {});
        json result = json::parse(response.output);
        REQUIRE(result.is_object());
        REQUIRE(result.contains("communities"));

        uint32_t han_comm = 0;
        uint32_t chew_comm = 0;
        uint32_t luke_comm = 0;
        uint32_t vader_comm = 0;

        for (auto& [id, members] : result["communities"].items()) {
            uint32_t comm_id = std::stoul(id);
            for (auto& member : members) {
                if (member["name"] == "Han Solo") han_comm = comm_id;
                if (member["name"] == "Chewbacca") chew_comm = comm_id;
                if (member["name"] == "Luke Skywalker") luke_comm = comm_id;
                if (member["name"] == "Darth Vader") vader_comm = comm_id;
            }
        }

        CHECK(han_comm == chew_comm);
        CHECK(luke_comm == vader_comm);
        CHECK(han_comm != luke_comm);
    }

    SECTION("Knowledge Graph Builds Multiple Communities from Text") {
        GraphAddon text_graph;
        std::string text =
            "NASA collaborates with SpaceX on Mars missions. "
            "NASA and SpaceX engineers review Mars launch plans together. "
            "CRISPR advances genomics research. "
            "DNA sequencing and CRISPR support genomics breakthroughs.";

        engine.build_knowledge_graph(text, text_graph, 15);
        text_graph.detect_communities(10);

        auto response = text_graph.process_impl("", {});
        REQUIRE(response.success);

        json result = json::parse(response.output);
        REQUIRE(result.is_object());
        REQUIRE(result.contains("communities"));
        REQUIRE(response.metrics.at("nodes") >= 5.0);

        bool found_space_cluster = false;
        bool found_bio_cluster = false;

        for (auto& [id, members] : result["communities"].items()) {
            bool has_nasa = false;
            bool has_spacex = false;
            bool has_mars = false;
            bool has_crispr = false;
            bool has_dna = false;
            bool has_genomics = false;

            for (auto& member : members) {
                const std::string name = member["name"];
                if (name == "NASA") has_nasa = true;
                if (name == "SpaceX") has_spacex = true;
                if (name == "Mars") has_mars = true;
                if (name == "CRISPR") has_crispr = true;
                if (name == "DNA") has_dna = true;
                if (name == "genomics") has_genomics = true;
            }

            if (has_nasa && has_spacex && has_mars) found_space_cluster = true;
            if (has_crispr && has_dna && has_genomics) found_bio_cluster = true;
        }

        CHECK(found_space_cluster);
        CHECK(found_bio_cluster);
    }
}
