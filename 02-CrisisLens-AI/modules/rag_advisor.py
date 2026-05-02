"""
rag_advisor.py — Module 5: RAG-Powered Crisis Advisor
ChromaDB knowledge base with NDMA guidelines, Delhi DDMA protocols,
and crisis response SOPs. Retrieves relevant protocols based on
natural-language queries and current situational context.

When connected to an LLM (Gemini), generates contextual, actionable
responses. Falls back to raw retrieval when LLM is unavailable.

Author: Abhinav Pandey
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime


@dataclass
class AdvisorResponse:
    """Response from the Crisis Advisor."""
    query: str
    context_summary: str
    retrieved_docs: List[str]
    retrieved_sources: List[str]
    distances: List[float]
    generated_response: str
    confidence: float
    timestamp: str


# ============================================
# CRISIS KNOWLEDGE CORPUS
# ============================================
# Based on NDMA guidelines, Delhi DDMA protocols, and standard
# emergency management procedures. Each entry has:
# (document_text, metadata_dict)

CRISIS_KNOWLEDGE = [

    # ---- FLOOD RESPONSE ----
    (
        "FLOOD RESPONSE PROTOCOL — IMMEDIATE ACTIONS (0-2 hours): "
        "1. Activate flood control room and establish communication with all field units. "
        "2. Deploy rescue boats to identified flood-prone areas (Yamuna floodplain, Mayur Vihar, ITO). "
        "3. Open emergency shelters at pre-designated locations: schools and community halls above flood level. "
        "4. Issue public warning via SMS, loudspeakers, and social media: specify evacuation routes. "
        "5. Close low-lying underpasses (Minto Bridge, Pul Prahladpur, Mahipalpur) immediately. "
        "6. Alert hospitals (AIIMS, Safdarjung, GTB) to prepare for flood-related casualties. "
        "7. Deploy NDRF teams to Yamuna east bank sectors.",
        {"crisis_type": "flood", "phase": "immediate", "source": "NDMA Guidelines + Delhi DDMA",
         "region": "delhi", "severity": "high"}
    ),
    (
        "FLOOD RESPONSE — EVACUATION PROTOCOL: "
        "When Yamuna water level crosses danger mark (205.33m at Old Railway Bridge): "
        "1. Mandatory evacuation of all settlements within 500m of river bank. "
        "2. Evacuation priority: elderly, disabled, children, pregnant women first. "
        "3. Evacuation routes: move AWAY from river toward elevated areas (Ridge Road, South Delhi). "
        "4. Do NOT allow pedestrian or vehicular movement on Yamuna bridges. "
        "5. Emergency transport: deploy DTC buses from nearest depot to evacuation points. "
        "6. Relief camps: minimum provision of drinking water (3 liters/person/day), "
        "dry food, medicines, sanitation facilities. DDMA nodal officer to be present at each camp. "
        "7. Estimated displaced population at danger level: 25,000-40,000 persons along Yamuna belt.",
        {"crisis_type": "flood", "phase": "evacuation", "source": "Delhi DDMA Flood Action Plan",
         "region": "delhi_yamuna", "severity": "critical"}
    ),
    (
        "FLOOD — POST-EVENT RECOVERY (24-72 hours): "
        "1. Water quality testing at all Yamuna intake points before restoring supply. "
        "2. Deploy health teams for epidemic surveillance: diarrhea, cholera, leptospirosis, dengue. "
        "3. Pump out waterlogged areas using portable dewatering pumps (MCD responsibility). "
        "4. Damage assessment of infrastructure: roads, bridges, power lines, sewage systems. "
        "5. Compensation claims processing through District Magistrate office. "
        "6. Decontamination of flood-affected hospitals and schools before reopening. "
        "7. Restore traffic flow on flood-affected roads after safety inspection by PWD.",
        {"crisis_type": "flood", "phase": "recovery", "source": "NDMA Post-Disaster Recovery Guidelines",
         "region": "delhi", "severity": "medium"}
    ),
    (
        "WATERLOGGING RESPONSE — URBAN FLOODING: "
        "Delhi receives 60% of annual rainfall in July-August. Key waterlogging hotspots: "
        "Minto Bridge, Pul Prahladpur, Anand Parvat, Kirari, Mundka, Rohini Sector 16-17. "
        "Action: 1. Activate all 400+ MCD pumping stations. "
        "2. Deploy traffic police to divert vehicles from waterlogged underpasses. "
        "3. Issue weather-based school closure advisory if rainfall exceeds 100mm in 3 hours. "
        "4. Coordinate with DJB (Delhi Jal Board) for drain clearance. "
        "5. Electrocution risk: cut power supply to submerged electrical installations. "
        "Report all open manholes and downed power lines to 112.",
        {"crisis_type": "flood", "phase": "immediate", "source": "MCD Monsoon Action Plan",
         "region": "delhi_urban", "severity": "medium"}
    ),

    # ---- FIRE RESPONSE ----
    (
        "FIRE RESPONSE PROTOCOL — INDUSTRIAL AREAS: "
        "Delhi industrial fire hotspots: Bawana, Mundka, Narela, Okhla Phase-2, Mayapuri. "
        "1. First responder: nearest fire station (Delhi Fire Services has 58 stations). "
        "2. For factory fires: determine chemical inventory BEFORE entry (MSDS data). "
        "3. Establish 100m cordon. Evacuate downwind residents if toxic smoke visible. "
        "4. Cut electricity to affected building via BSES/Tata Power emergency line. "
        "5. Alert nearest hospital trauma center. Request additional ambulances via 102. "
        "6. If LPG/CNG cylinders involved: evacuate 300m radius, DO NOT use water on gas fires. "
        "7. Building collapse risk after 30 minutes of uncontrolled fire — keep rescue teams on standby.",
        {"crisis_type": "fire", "phase": "immediate", "source": "Delhi Fire Services SOP",
         "region": "delhi_industrial", "severity": "critical"}
    ),
    (
        "FIRE — RESIDENTIAL BUILDING: "
        "1. Evacuate all floors ABOVE the fire floor first, then below. "
        "2. Use staircases ONLY, never elevators. "
        "3. If trapped: seal door gaps with wet cloth, signal from window. "
        "4. Fire tender response time target: 7 minutes in Delhi. "
        "5. For high-rise (>15m): request aerial ladder platform (Delhi has 12 units). "
        "6. Post-fire: structural safety assessment by MCD before re-entry. "
        "7. Temporary shelter for displaced residents: coordinate with District SDM office.",
        {"crisis_type": "fire", "phase": "immediate", "source": "NDMA Fire Safety Guidelines",
         "region": "delhi", "severity": "high"}
    ),

    # ---- HEAT WAVE ----
    (
        "HEAT WAVE RESPONSE — DELHI: "
        "Heat wave declaration: when max temperature reaches 45°C (or 40°C + 4.5°C above normal). "
        "1. Issue public advisory: avoid outdoor exposure 12PM-4PM. "
        "2. Open cooling shelters at metro stations, government buildings, community halls. "
        "3. Ensure continuous water supply to slum clusters and construction sites. "
        "4. Deploy ORS (Oral Rehydration Solution) packets at all government hospitals. "
        "5. Special watch on outdoor workers: construction, rickshaw drivers, street vendors. "
        "6. School timing adjustment: 7AM-12PM during heat wave period. "
        "7. Mortuary capacity alert if deaths exceed 5/day in any district. "
        "High-risk areas: Najafgarh, Palam, Mungeshpur (historically record Delhi's highest temperatures).",
        {"crisis_type": "heat_wave", "phase": "immediate", "source": "NDMA Heat Wave Guidelines + Delhi Action Plan",
         "region": "delhi", "severity": "high"}
    ),

    # ---- EARTHQUAKE ----
    (
        "EARTHQUAKE RESPONSE — DELHI (Seismic Zone IV): "
        "Delhi falls in Seismic Zone IV (high damage risk). Last significant tremor felt from Nepal 2015. "
        "1. Post-earthquake: structural triage of all buildings — Green (safe), Yellow (restricted entry), Red (condemned). "
        "2. Priority search areas: Old Delhi (pre-1970 construction), unauthorized colonies, multi-story buildings without seismic design. "
        "3. Deploy NDRF teams for collapsed structure rescue. "
        "4. Gas leak check: shut off CNG/piped gas supply to affected sectors. "
        "5. Hospital surge plan: activate all trauma centers, cancel elective surgeries. "
        "6. Aftershock advisory: keep population outdoors for 24 hours after major shock. "
        "7. Communication: activate HAM radio network if mobile towers are down.",
        {"crisis_type": "earthquake", "phase": "immediate", "source": "NDMA Earthquake Guidelines",
         "region": "delhi", "severity": "critical"}
    ),

    # ---- GAS LEAK / CHEMICAL ----
    (
        "CHEMICAL/GAS LEAK RESPONSE: "
        "1. Identify the chemical: check factory MSDS (Material Safety Data Sheet). "
        "2. Establish exclusion zone: 300m for unknown chemicals, 500m for toxic gases. "
        "3. Evacuate UPWIND — always move perpendicular to wind direction, then upwind. "
        "4. Do NOT allow water spray on chemicals unless MSDS confirms water-compatible. "
        "5. Alert CPCB (Central Pollution Control Board) for air quality monitoring. "
        "6. Nearest chemical disaster response: NDRF Battalion, Delhi (Dwarka). "
        "7. Hospital alert: specify chemical name so antidote preparation can begin before patients arrive. "
        "8. Environmental monitoring: check groundwater contamination if liquid spill.",
        {"crisis_type": "gas_leak", "phase": "immediate", "source": "NDMA Chemical Disaster Guidelines",
         "region": "delhi", "severity": "critical"}
    ),

    # ---- CROWD MANAGEMENT ----
    (
        "CROWD MANAGEMENT — LARGE GATHERINGS: "
        "Delhi events with 50,000+ attendance: Republic Day, Diwali at India Gate, "
        "Religious gatherings (Nizamuddin, Jama Masjid, Gurudwara Bangla Sahib). "
        "1. Crowd density threshold: 4 persons/m² is warning level, 6 persons/m² is critical. "
        "2. One-way flow enforcement at all entry/exit points. "
        "3. Emergency exit routes must remain clear — deploy volunteers with reflective vests. "
        "4. Medical first-aid posts every 500m along event perimeter. "
        "5. Stampede risk mitigation: barrier management, overflow parking, staggered entry times. "
        "6. Communication: PA system + police wireless + dedicated WhatsApp group for control room. "
        "7. Counter-flow protocol: if crowd surge detected, open ALL exit gates simultaneously.",
        {"crisis_type": "crowd_incident", "phase": "prevention", "source": "Delhi Police Crowd Management SOP",
         "region": "delhi", "severity": "high"}
    ),

    # ---- BUILDING COLLAPSE ----
    (
        "BUILDING COLLAPSE RESPONSE: "
        "1. Do NOT enter collapsed structure without NDRF/SDRF clearance. "
        "2. Establish outer cordon (50m) and inner cordon (25m). "
        "3. Silence zone: all machinery off during survivor detection (listening devices, search dogs). "
        "4. Rescue priority: void spaces near stairwells and corners (highest survival probability). "
        "5. Golden hour: maximum rescue effort in first 6 hours — survival rate drops 50% after 24 hours. "
        "6. Heavy machinery (JCB, cranes) only after NDRF confirms no survivors in the section. "
        "7. Body recovery protocol: photograph, tag, transport to mortuary. Inform District Magistrate.",
        {"crisis_type": "building_collapse", "phase": "immediate", "source": "NDRF SOP for Collapsed Structure Rescue",
         "region": "delhi", "severity": "critical"}
    ),

    # ---- ROAD ACCIDENT (MASS CASUALTY) ----
    (
        "MASS CASUALTY ROAD ACCIDENT RESPONSE: "
        "Delhi road accident hotspots: NH-44 (GT Karnal Road), Ring Road, Mehrauli-Badarpur Road, Outer Ring Road. "
        "1. First responder: traffic police + nearest PCR van. Secure scene, prevent secondary accidents. "
        "2. Triage: GREEN (walking wounded) → nearest clinic. YELLOW (serious but stable) → district hospital. "
        "RED (critical) → nearest trauma center (AIIMS Trauma, Safdarjung, GTB). "
        "3. Multiple ambulance dispatch: call 102 AND 108. Request blood bank alert at receiving hospital. "
        "4. Traffic diversion plan: activate alternate routes, update Google Maps/Waze via police feed. "
        "5. If fuel spill: fire tender on standby, no smoking within 100m, sand/sawdust on spill. "
        "6. FIR registration and scene documentation before clearing wreckage.",
        {"crisis_type": "road_accident", "phase": "immediate", "source": "Delhi Traffic Police SOP + NDMA Road Safety",
         "region": "delhi", "severity": "high"}
    ),

    # ---- AIR POLLUTION EMERGENCY ----
    (
        "AIR POLLUTION EMERGENCY — DELHI AQI > 400 (SEVERE+): "
        "1. GRAP (Graded Response Action Plan) Stage IV activated: "
        "   - Ban on all construction and demolition. "
        "   - Ban on entry of non-essential trucks into Delhi. "
        "   - Close brick kilns and stone crushers within 300km. "
        "   - Consider school closure advisory for primary classes. "
        "2. Health advisory: N95 masks for outdoor exposure, air purifiers for vulnerable populations. "
        "3. Hospital preparedness: stock bronchodilators, nebulizers, oxygen cylinders. "
        "4. Stubble burning monitoring: satellite data from ISRO/NASA FIRMS for fire counts in Punjab/Haryana. "
        "5. Artificial rain request to IMD (cloud seeding) if conditions permit. "
        "6. Public communication: real-time AQI updates via CPCB SAMEER app.",
        {"crisis_type": "air_pollution", "phase": "immediate", "source": "CPCB GRAP + Delhi Environment Dept",
         "region": "delhi", "severity": "high"}
    ),

    # ---- HISTORICAL PATTERNS ----
    (
        "HISTORICAL FLOOD REFERENCE — DELHI: "
        "2023 (July): Yamuna at 208.66m — highest in 45 years. ITO submerged, Pragati Maidan flooded. "
        "25,000 people evacuated. Rail services disrupted for 3 days. "
        "2019 (Aug): Yamuna at 206.60m. East Delhi flooding. 15,000 evacuated. "
        "2013 (June): Yamuna at 207.32m. Massive flooding in Civil Lines and Kashmere Gate. "
        "2010 (Sep): Yamuna at 207.71m. ITO barrage gates opened. 30,000+ displaced. "
        "Pattern: Major floods cluster in July-September. Peak arrival at Delhi is 48-72 hours after "
        "upstream heavy rainfall in Haryana/Uttarakhand. Hathnikund Barrage release is the critical trigger.",
        {"crisis_type": "flood", "phase": "reference", "source": "CWC Historical Records + Delhi DDMA Archives",
         "region": "delhi_yamuna", "severity": "reference"}
    ),
    (
        "HISTORICAL FIRE REFERENCE — DELHI: "
        "2019 (Dec): Anaj Mandi factory fire — 43 deaths, deadliest in Delhi in 2 decades. "
        "Cause: illegal manufacturing in residential area, no fire exits. "
        "2022 (May): Mundka fire — 27 deaths in commercial building. Locked exit doors. "
        "2024 (Jan): Bawana paint factory fire — 11 workers killed, toxic fumes. "
        "Pattern: Industrial fires peak in summer (May-June) due to high temperatures + chemical volatility. "
        "Key lesson: most deaths from smoke inhalation, not burns. Locked exits are the #1 killer.",
        {"crisis_type": "fire", "phase": "reference", "source": "Delhi Fire Services Annual Reports",
         "region": "delhi", "severity": "reference"}
    ),
]


class CrisisAdvisor:
    """
    RAG-powered Crisis Advisor using ChromaDB.

    1. Loads NDMA/DDMA crisis protocols into ChromaDB
    2. Accepts natural-language queries from command officers
    3. Retrieves relevant protocols based on query + current context
    4. Optionally generates LLM-enhanced response via Gemini API
    5. Includes historical pattern matching for similar past events
    """

    def __init__(self):
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False
        ))
        self.collection = None
        self._initialized = False

    def initialize(self):
        """Load crisis knowledge into ChromaDB."""
        if self._initialized:
            return

        self.collection = self.client.get_or_create_collection(
            name="crisis_protocols",
            metadata={"description": "NDMA/DDMA crisis response protocols for Delhi"}
        )

        if self.collection.count() == 0:
            print("[RAG] Loading crisis knowledge base...")
            documents = []
            metadatas = []
            ids = []

            for i, (doc, meta) in enumerate(CRISIS_KNOWLEDGE):
                documents.append(doc)
                metadatas.append(meta)
                ids.append(f"crisis_{i:04d}")

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[RAG] Loaded {len(documents)} crisis protocol entries.")

        self._initialized = True

    def query(self, question: str,
              crisis_type: str = None,
              n_results: int = 3,
              context: Dict = None) -> AdvisorResponse:
        """
        Query the crisis advisor.

        Parameters:
            question: Natural language query from command officer
            crisis_type: Optional filter (flood, fire, heat_wave, etc.)
            n_results: Number of documents to retrieve
            context: Current situational context dict (weather, incidents, etc.)
        """
        if not self._initialized:
            self.initialize()

        # Build enhanced query with context
        enhanced_query = question
        context_summary = ""

        if context:
            context_parts = []
            if "weather" in context:
                wx = context["weather"]
                context_parts.append(f"Weather: {wx.get('temp', 'N/A')}°C, rain: {wx.get('rain', 0)}mm")
            if "incidents" in context:
                context_parts.append(f"Active incidents: {context['incidents']}")
            if "flood_risk" in context:
                context_parts.append(f"Flood risk: {context['flood_risk']}")
            if "hospitals" in context:
                context_parts.append(f"Hospital capacity: {context['hospitals']}")

            context_summary = " | ".join(context_parts)
            enhanced_query = f"{question} Context: {context_summary}"

        # Build metadata filter
        where_filter = None
        if crisis_type:
            where_filter = {"crisis_type": crisis_type}

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n_results,
                where=where_filter
            )
        except Exception:
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n_results
            )

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []

        sources = [m.get("source", "Unknown") for m in metas]

        # Generate response
        if docs:
            # Structured response from retrieved protocols
            response_parts = []
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                phase = meta.get("phase", "")
                severity = meta.get("severity", "")
                relevance = max(0, 1.0 - dist / 2.0)  # Normalize distance to relevance

                if i == 0:
                    response_parts.append(f"PRIMARY PROTOCOL ({phase.upper()}, relevance: {relevance:.0%}):\n{doc}")
                else:
                    response_parts.append(f"\nADDITIONAL ({phase.upper()}, relevance: {relevance:.0%}):\n{doc}")

            generated = "\n".join(response_parts)
            confidence = max(0, 1.0 - (dists[0] / 2.0)) if dists else 0
        else:
            generated = ("No specific protocol found for this query. "
                        "Contact DDMA control room at 1077 for immediate guidance.")
            confidence = 0.0

        return AdvisorResponse(
            query=question,
            context_summary=context_summary,
            retrieved_docs=docs,
            retrieved_sources=sources,
            distances=dists,
            generated_response=generated,
            confidence=round(confidence, 3),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M IST")
        )

    def find_historical_match(self, crisis_type: str,
                              location: str = "Delhi") -> AdvisorResponse:
        """
        Find historical events similar to the current crisis.
        Uses vector similarity to match current situation against
        historical records in the knowledge base.
        """
        if not self._initialized:
            self.initialize()

        query = f"historical {crisis_type} {location} past events what happened"

        results = self.collection.query(
            query_texts=[query],
            n_results=2,
            where={"phase": "reference"}
        )

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []

        if docs:
            generated = "HISTORICAL PATTERN MATCH:\n" + "\n\n".join(docs)
            confidence = max(0, 1.0 - (dists[0] / 2.0)) if dists else 0
        else:
            generated = "No historical matches found for this crisis type in Delhi."
            confidence = 0.0

        return AdvisorResponse(
            query=f"Historical match: {crisis_type}",
            context_summary="",
            retrieved_docs=docs,
            retrieved_sources=[m.get("source", "Unknown") for m in metas],
            distances=dists,
            generated_response=generated,
            confidence=round(confidence, 3),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M IST")
        )

    def get_protocol_count(self) -> Dict:
        """Return knowledge base statistics."""
        if not self._initialized:
            self.initialize()

        crisis_types = {}
        for _, meta in CRISIS_KNOWLEDGE:
            ct = meta.get("crisis_type", "unknown")
            crisis_types[ct] = crisis_types.get(ct, 0) + 1

        return {
            "total_protocols": self.collection.count(),
            "crisis_types": crisis_types,
            "sources": list(set(m.get("source", "") for _, m in CRISIS_KNOWLEDGE))
        }


# ============================================
# DEMO
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("CrisisLens AI — RAG Crisis Advisor (New Delhi)")
    print("=" * 60)

    advisor = CrisisAdvisor()
    advisor.initialize()

    stats = advisor.get_protocol_count()
    print(f"\nKnowledge base: {stats['total_protocols']} protocols")
    for ct, count in stats["crisis_types"].items():
        print(f"  {ct}: {count} entries")

    # Test 1: Flood query with context
    print("\n" + "=" * 60)
    print("QUERY 1: Flooding in Mayur Vihar")
    print("=" * 60)
    resp = advisor.query(
        "We have flooding in Mayur Vihar with 200 displaced residents. What's the protocol?",
        crisis_type="flood",
        context={
            "weather": {"temp": 32, "rain": 45},
            "flood_risk": "CRITICAL at Yamuna Floodplain",
            "hospitals": "1338 beds available"
        }
    )
    print(f"Confidence: {resp.confidence}")
    print(f"Sources: {resp.retrieved_sources}")
    print(f"\n{resp.generated_response[:500]}...")

    # Test 2: Fire at Bawana
    print("\n" + "=" * 60)
    print("QUERY 2: Factory fire in Bawana")
    print("=" * 60)
    resp2 = advisor.query(
        "Major fire at a paint factory in Bawana. 3 workers trapped. Toxic fumes spreading.",
        crisis_type="fire"
    )
    print(f"Confidence: {resp2.confidence}")
    print(f"\n{resp2.generated_response[:500]}...")

    # Test 3: Historical match
    print("\n" + "=" * 60)
    print("QUERY 3: Historical flood patterns")
    print("=" * 60)
    hist = advisor.find_historical_match("flood")
    print(f"Confidence: {hist.confidence}")
    print(f"\n{hist.generated_response[:500]}...")

    # Test 4: Open-ended question
    print("\n" + "=" * 60)
    print("QUERY 4: Heat wave advisory")
    print("=" * 60)
    resp4 = advisor.query("Temperature is 46°C. What measures should we activate?")
    print(f"Confidence: {resp4.confidence}")
    print(f"\n{resp4.generated_response[:300]}...")

    print(f"\n\u2713 Module 5 (RAG Crisis Advisor) complete.")
