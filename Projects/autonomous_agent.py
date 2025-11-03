"""
Autonomous AI Agent Framework
Author: Cazandra Aporbo
Upload Date: November 2025

autonomous agent system that combines multiple AI paradigms into a 
unified cognitive architecture. features self-directed learning, 
goal decomposition, tool use, memory systems, and  metacognitive 
reasoning capabilities.

inspiration from cognitive science, Global Workspace Theory and 
dual-process theories of reasoning.

"""

import asyncio
import inspect
import json
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union,
    AsyncIterator, Protocol, runtime_checkable, TypeVar, Generic
)
import hashlib
import pickle
import sqlite3
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

# Advanced packages that provide sophisticated functionality
try:
    import attrs  # Better than dataclasses with validators and converters
    from attrs import define, field as attr_field, validators
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'attrs'])
    import attrs
    from attrs import define, field as attr_field, validators

try:
    import networkx as nx  # Graph algorithms for knowledge representation
    from networkx.algorithms import shortest_path, pagerank, community
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'networkx'])
    import networkx as nx
    from networkx.algorithms import shortest_path, pagerank, community

try:
    from pyrsistent import pmap, pvector, pset  # Immutable data structures
    from pyrsistent import PMap, PVector, PSet
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'pyrsistent'])
    from pyrsistent import pmap, pvector, pset
    from pyrsistent import PMap, PVector, PSet

try:
    from transitions import Machine  # State machine for agent behavior
    from transitions.extensions import GraphMachine
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'transitions'])
    from transitions import Machine
    from transitions.extensions import GraphMachine

try:
    import funcy  # Functional programming utilities
    from funcy import (
        memoize, cached_property, retry, collecting, 
        joining, merge, project, omit, walk_values
    )
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'funcy'])
    import funcy
    from funcy import (
        memoize, cached_property, retry, collecting,
        joining, merge, project, omit, walk_values
    )

try:
    from toolz import pipe, compose, curry, partial  # Functional pipelines
    from toolz.functoolz import do, excepts, flip
    from toolz.itertoolz import sliding_window, partition
    from toolz.dicttoolz import merge_with, valmap, keymap
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'toolz'])
    from toolz import pipe, compose, curry, partial
    from toolz.functoolz import do, excepts, flip
    from toolz.itertoolz import sliding_window, partition
    from toolz.dicttoolz import merge_with, valmap, keymap

try:
    from loguru import logger  # Advanced logging with automatic rotation
    logger.add("agent_{time}.log", rotation="500 MB", retention="10 days", level="DEBUG")
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'loguru'])
    from loguru import logger
    logger.add("agent_{time}.log", rotation="500 MB", retention="10 days", level="DEBUG")

try:
    from tenacity import (  # Advanced retry logic with backoff
        retry, stop_after_attempt, wait_exponential,
        retry_if_exception_type, before_sleep_log
    )
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'tenacity'])
    from tenacity import (
        retry, stop_after_attempt, wait_exponential,
        retry_if_exception_type, before_sleep_log
    )

try:
    from cachetools import TTLCache, LFUCache, cached  # Advanced caching
    from cachetools.keys import hashkey
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'cachetools'])
    from cachetools import TTLCache, LFUCache, cached
    from cachetools.keys import hashkey

try:
    import msgspec  # Fast serialization, better than pickle/json
    from msgspec import Struct, field as msgspec_field
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'msgspec'])
    import msgspec
    from msgspec import Struct, field as msgspec_field

try:
    from rich.console import Console  # Beautiful terminal output
    from rich.table import Table
    from rich.panel import Panel
    from rich.tree import Tree
    console = Console()
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'rich'])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.tree import Tree
    console = Console()

try:
    import httpx  # Modern async HTTP client
    from httpx import AsyncClient
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'httpx'])
    import httpx
    from httpx import AsyncClient


T = TypeVar('T')
AgentType = TypeVar('AgentType', bound='BaseAgent')


class ThoughtType(Enum):
    """Categories of cognitive processes the agent can engage in."""
    OBSERVATION = auto()      # Perceiving and noting facts
    HYPOTHESIS = auto()       # Forming testable explanations
    PLANNING = auto()         # Creating action sequences
    REFLECTION = auto()       # Analyzing own thoughts and actions
    ABSTRACTION = auto()      # Extracting general principles
    ANALOGY = auto()          # Finding similar patterns
    COUNTERFACTUAL = auto()   # Considering alternatives
    METACOGNITION = auto()    # Thinking about thinking


class MemoryType(Enum):
    """Different memory systems in the cognitive architecture."""
    SENSORY = auto()          # Very short-term sensory buffer
    WORKING = auto()          # Active manipulation of information
    EPISODIC = auto()         # Specific experiences and events
    SEMANTIC = auto()         # General knowledge and facts
    PROCEDURAL = auto()       # Skills and how-to knowledge
    PROSPECTIVE = auto()      # Future intentions and plans


@define
class Thought:
    """
    Represents a single cognitive unit in the agent's thinking process.
    
    Thoughts are immutable and form chains of reasoning. Each thought
    has a type, content, confidence level, and can reference other thoughts
    to form complex reasoning graphs.
    """
    id: str = attr_field(factory=lambda: str(uuid.uuid4()))
    type: ThoughtType = attr_field(validator=validators.instance_of(ThoughtType))
    content: Any = attr_field()
    confidence: float = attr_field(validator=validators.instance_of(float))
    timestamp: datetime = attr_field(factory=datetime.now)
    parent_thoughts: List[str] = attr_field(factory=list)
    metadata: Dict[str, Any] = attr_field(factory=dict)
    
    @confidence.validator
    def check_confidence(self, attribute, value):
        """Ensure confidence is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize thought to dictionary."""
        return {
            'id': self.id,
            'type': self.type.name,
            'content': self.content,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'parent_thoughts': self.parent_thoughts,
            'metadata': self.metadata
        }


@define
class Memory:
    """
    Represents a memory unit with decay and reinforcement mechanisms.
    
    Memories have strength that decays over time following a forgetting curve,
    but can be reinforced through retrieval. This models human-like memory
    dynamics including primacy, recency, and spacing effects.
    """
    id: str = attr_field(factory=lambda: str(uuid.uuid4()))
    type: MemoryType = attr_field(validator=validators.instance_of(MemoryType))
    content: Any = attr_field()
    strength: float = attr_field(default=1.0)
    last_accessed: datetime = attr_field(factory=datetime.now)
    access_count: int = attr_field(default=0)
    creation_time: datetime = attr_field(factory=datetime.now)
    associations: Set[str] = attr_field(factory=set)
    context: Dict[str, Any] = attr_field(factory=dict)
    
    def decay(self, current_time: datetime, decay_rate: float = 0.1) -> float:
        """
        Calculate memory strength after decay.
        
        Uses Ebbinghaus forgetting curve with modifications for
        reinforcement through retrieval practice.
        """
        time_elapsed = (current_time - self.last_accessed).total_seconds() / 3600
        
        # Modified forgetting curve with reinforcement factor
        reinforcement_factor = 1 + (0.1 * self.access_count)
        decayed_strength = self.strength * (reinforcement_factor ** (-decay_rate * time_elapsed))
        
        return max(0.01, min(1.0, decayed_strength))  # Clamp between 0.01 and 1.0
    
    def reinforce(self, amount: float = 0.1) -> None:
        """Strengthen memory through retrieval or rehearsal."""
        self.strength = min(1.0, self.strength + amount)
        self.access_count += 1
        self.last_accessed = datetime.now()


class Goal(msgspec.Struct):
    """
    Represents an agent goal with hierarchical decomposition.
    
    Goals can be decomposed into subgoals, have priorities, deadlines,
    and success criteria. The agent uses these to direct its behavior
    and evaluate progress.
    """
    id: str = msgspec_field(default_factory=lambda: str(uuid.uuid4()))
    description: str = msgspec_field()
    priority: float = msgspec_field(default=0.5)
    deadline: Optional[datetime] = msgspec_field(default=None)
    parent_goal: Optional[str] = msgspec_field(default=None)
    subgoals: List[str] = msgspec_field(default_factory=list)
    success_criteria: Dict[str, Any] = msgspec_field(default_factory=dict)
    status: str = msgspec_field(default="pending")
    progress: float = msgspec_field(default=0.0)
    
    def is_achievable(self, resources: Dict[str, Any]) -> bool:
        """Evaluate if goal is achievable with available resources."""
        # Simplified check - in practice would involve complex reasoning
        required_resources = self.success_criteria.get('required_resources', {})
        return all(
            resources.get(resource, 0) >= amount
            for resource, amount in required_resources.items()
        )
    
    def update_progress(self, achievement: Dict[str, Any]) -> None:
        """Update goal progress based on achievements."""
        if not self.success_criteria:
            return
        
        completed_criteria = sum(
            1 for criterion, target in self.success_criteria.items()
            if achievement.get(criterion, 0) >= target
        )
        
        self.progress = completed_criteria / len(self.success_criteria)
        
        if self.progress >= 1.0:
            self.status = "completed"
        elif self.progress > 0:
            self.status = "in_progress"


@runtime_checkable
class Tool(Protocol):
    """Protocol defining the interface for agent tools."""
    
    @property
    def name(self) -> str:
        """Tool name for invocation."""
        ...
    
    @property
    def description(self) -> str:
        """Human-readable description of tool functionality."""
        ...
    
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool with given arguments."""
        ...
    
    def validate_args(self, *args: Any, **kwargs: Any) -> bool:
        """Validate arguments before execution."""
        ...


class WebSearchTool:
    """Tool for searching the web and extracting information."""
    
    name = "web_search"
    description = "Search the web for information on any topic"
    
    def __init__(self):
        self.client = AsyncClient(timeout=30.0)
        self.cache = TTLCache(maxsize=100, ttl=3600)
    
    def validate_args(self, query: str, max_results: int = 5) -> bool:
        """Validate search parameters."""
        return isinstance(query, str) and len(query) > 0 and 0 < max_results <= 20
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def execute(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Execute web search and return results.
        
        In a real implementation, this would call an actual search API.
        Here we simulate with mock data for demonstration.
        """
        cache_key = hashkey(query, max_results)
        
        if cache_key in self.cache:
            logger.debug(f"Returning cached results for query: {query}")
            return self.cache[cache_key]
        
        # Simulated search results
        await asyncio.sleep(0.5)  # Simulate network delay
        
        results = [
            {
                "title": f"Result {i+1} for {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a relevant snippet about {query}..."
            }
            for i in range(max_results)
        ]
        
        self.cache[cache_key] = results
        return results


class CalculatorTool:
    """Tool for performing mathematical calculations."""
    
    name = "calculator"
    description = "Perform mathematical calculations and symbolic reasoning"
    
    def validate_args(self, expression: str) -> bool:
        """Validate mathematical expression."""
        # Check for potentially dangerous operations
        forbidden = ['__', 'import', 'exec', 'eval', 'open', 'file']
        return not any(f in expression for f in forbidden)
    
    async def execute(self, expression: str) -> float:
        """
        Safely evaluate mathematical expression.
        
        Uses a restricted evaluation environment to prevent code injection.
        """
        if not self.validate_args(expression):
            raise ValueError("Invalid or unsafe expression")
        
        # Safe math operations only
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'len': len,
            '__builtins__': {}
        }
        
        try:
            # Add math module functions
            import math
            safe_dict.update({
                name: getattr(math, name)
                for name in dir(math)
                if not name.startswith('_')
            })
            
            result = eval(expression, safe_dict)
            return float(result)
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            raise ValueError(f"Cannot evaluate expression: {e}")


class KnowledgeGraph:
    """
    Knowledge representation using a directed graph structure.
    
    Nodes represent concepts, edges represent relationships.
    Supports reasoning through graph algorithms like shortest path,
    PageRank for importance, and community detection for clustering.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._lock = threading.Lock()
        
    def add_concept(self, concept: str, attributes: Dict[str, Any] = None) -> None:
        """Add a concept node to the knowledge graph."""
        with self._lock:
            self.graph.add_node(concept, **(attributes or {}))
            logger.debug(f"Added concept: {concept}")
    
    def add_relation(self, source: str, target: str, relation_type: str, 
                     weight: float = 1.0) -> None:
        """Add a directed relationship between concepts."""
        with self._lock:
            self.graph.add_edge(source, target, 
                              relation=relation_type, weight=weight)
            logger.debug(f"Added relation: {source} -{relation_type}-> {target}")
    
    @lru_cache(maxsize=128)
    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two concepts."""
        try:
            return shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def get_related_concepts(self, concept: str, max_distance: int = 2) -> Set[str]:
        """Get concepts within a certain distance from given concept."""
        if concept not in self.graph:
            return set()
        
        related = set()
        current_level = {concept}
        
        for _ in range(max_distance):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.neighbors(node))
                next_level.update(self.graph.predecessors(node))
            related.update(next_level)
            current_level = next_level
        
        related.discard(concept)  # Remove the original concept
        return related
    
    def compute_importance(self) -> Dict[str, float]:
        """Compute importance of concepts using PageRank."""
        if len(self.graph) == 0:
            return {}
        return pagerank(self.graph)
    
    def find_communities(self) -> List[Set[str]]:
        """Detect communities of related concepts."""
        if len(self.graph) == 0:
            return []
        
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        communities = community.greedy_modularity_communities(undirected)
        return [set(c) for c in communities]


class ReasoningEngine:
    """
    Implements various reasoning strategies including deduction, induction,
    abduction, and analogical reasoning. Uses a combination of symbolic
    and probabilistic approaches.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge = knowledge_graph
        self.inference_cache = LFUCache(maxsize=256)
    
    def deduce(self, premises: List[Thought], rules: List[Callable]) -> List[Thought]:
        """
        Apply deductive reasoning using given premises and rules.
        
        Implements a simple forward chaining algorithm to derive
        new conclusions from premises.
        """
        conclusions = []
        premise_contents = [p.content for p in premises]
        
        for rule in rules:
            try:
                # Apply rule to premises
                result = rule(premise_contents)
                if result is not None:
                    conclusion = Thought(
                        type=ThoughtType.HYPOTHESIS,
                        content=result,
                        confidence=min(p.confidence for p in premises) * 0.9,
                        parent_thoughts=[p.id for p in premises],
                        metadata={'reasoning': 'deduction', 'rule': rule.__name__}
                    )
                    conclusions.append(conclusion)
            except Exception as e:
                logger.debug(f"Rule {rule.__name__} not applicable: {e}")
        
        return conclusions
    
    def induce(self, observations: List[Thought], min_support: float = 0.6) -> List[Thought]:
        """
        Apply inductive reasoning to find patterns in observations.
        
        Uses frequency analysis and pattern mining to generate
        general hypotheses from specific observations.
        """
        patterns = defaultdict(int)
        total_observations = len(observations)
        
        if total_observations < 2:
            return []
        
        # Extract patterns from observations
        for obs in observations:
            if isinstance(obs.content, dict):
                for key, value in obs.content.items():
                    patterns[(key, value)] += 1
        
        # Generate hypotheses for frequent patterns
        hypotheses = []
        for (key, value), count in patterns.items():
            support = count / total_observations
            
            if support >= min_support:
                hypothesis = Thought(
                    type=ThoughtType.HYPOTHESIS,
                    content=f"Pattern: {key} tends to be {value}",
                    confidence=support,
                    parent_thoughts=[obs.id for obs in observations[:3]],  # Link to first few
                    metadata={'reasoning': 'induction', 'support': support}
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def abduce(self, observation: Thought, knowledge_base: List[Thought]) -> List[Thought]:
        """
        Apply abductive reasoning to find best explanations.
        
        Generates plausible explanations for observations based on
        existing knowledge, ranked by likelihood.
        """
        explanations = []
        
        for knowledge in knowledge_base:
            if knowledge.type == ThoughtType.HYPOTHESIS:
                # Check if knowledge could explain observation
                similarity = self._compute_similarity(observation.content, knowledge.content)
                
                if similarity > 0.5:
                    explanation = Thought(
                        type=ThoughtType.HYPOTHESIS,
                        content=f"Explanation: {knowledge.content} explains {observation.content}",
                        confidence=similarity * knowledge.confidence,
                        parent_thoughts=[observation.id, knowledge.id],
                        metadata={'reasoning': 'abduction', 'similarity': similarity}
                    )
                    explanations.append(explanation)
        
        # Sort by confidence and return top explanations
        explanations.sort(key=lambda x: x.confidence, reverse=True)
        return explanations[:3]
    
    def find_analogy(self, source: Thought, target_domain: str) -> Optional[Thought]:
        """
        Find analogical mappings between source and target domains.
        
        Uses structural mapping to find similar relationships in
        different domains.
        """
        source_concepts = self.knowledge.get_related_concepts(
            str(source.content), max_distance=1
        )
        
        target_concepts = self.knowledge.get_related_concepts(
            target_domain, max_distance=2
        )
        
        if not source_concepts or not target_concepts:
            return None
        
        # Find structural similarity
        mapping_score = len(source_concepts.intersection(target_concepts)) / \
                       max(len(source_concepts), len(target_concepts))
        
        if mapping_score > 0.3:
            analogy = Thought(
                type=ThoughtType.ANALOGY,
                content=f"Analogy: {source.content} is like {target_domain}",
                confidence=mapping_score,
                parent_thoughts=[source.id],
                metadata={'reasoning': 'analogy', 'mapping_score': mapping_score}
            )
            return analogy
        
        return None
    
    def _compute_similarity(self, content1: Any, content2: Any) -> float:
        """
        Compute similarity between two pieces of content.
        
        In a real implementation, this would use embeddings or
        more sophisticated similarity measures.
        """
        if isinstance(content1, str) and isinstance(content2, str):
            # Simple character-level similarity
            common = len(set(content1.lower().split()) & set(content2.lower().split()))
            total = max(len(content1.split()), len(content2.split()))
            return common / total if total > 0 else 0
        
        return 0.5  # Default similarity for unknown types


class CognitiveArchitecture:
    """
    The main cognitive architecture that orchestrates all components.
    
    Implements a Global Workspace Theory-inspired architecture where
    different modules compete for access to a global workspace for
    conscious processing.
    """
    
    def __init__(self):
        self.working_memory = deque(maxlen=7)  # Miller's magical number
        self.long_term_memory = {}  # Type -> List[Memory]
        self.knowledge_graph = KnowledgeGraph()
        self.reasoning_engine = ReasoningEngine(self.knowledge_graph)
        self.attention_focus = None
        self.current_goals = []
        self.thought_stream = []
        
        # Initialize memory systems
        for memory_type in MemoryType:
            self.long_term_memory[memory_type] = []
    
    def perceive(self, stimulus: Dict[str, Any]) -> Thought:
        """
        Process incoming stimulus and create observation thought.
        
        Applies attention filtering and salience detection to determine
        what aspects of the stimulus to focus on.
        """
        # Compute salience based on current goals and past experience
        salience = self._compute_salience(stimulus)
        
        observation = Thought(
            type=ThoughtType.OBSERVATION,
            content=stimulus,
            confidence=salience,
            metadata={'perceived_at': datetime.now().isoformat()}
        )
        
        # Add to working memory if salient enough
        if salience > 0.3:
            self.working_memory.append(observation)
            self.thought_stream.append(observation)
        
        # Store in sensory memory
        self._store_memory(MemoryType.SENSORY, observation.content)
        
        return observation
    
    def think(self, depth: int = 3) -> List[Thought]:
        """
        Engage in deliberate thinking process.
        
        Combines different reasoning strategies to generate new thoughts
        from current working memory contents.
        """
        if not self.working_memory:
            return []
        
        thoughts_generated = []
        current_thoughts = list(self.working_memory)
        
        for _ in range(depth):
            # Apply different reasoning strategies
            new_thoughts = []
            
            # Deductive reasoning
            rules = self._get_applicable_rules()
            deductions = self.reasoning_engine.deduce(current_thoughts, rules)
            new_thoughts.extend(deductions)
            
            # Inductive reasoning on observations
            observations = [t for t in current_thoughts if t.type == ThoughtType.OBSERVATION]
            if len(observations) >= 3:
                inductions = self.reasoning_engine.induce(observations)
                new_thoughts.extend(inductions)
            
            # Abductive reasoning for explanations
            for thought in current_thoughts:
                if thought.type == ThoughtType.OBSERVATION:
                    knowledge_base = self._retrieve_relevant_knowledge(thought)
                    explanations = self.reasoning_engine.abduce(thought, knowledge_base)
                    new_thoughts.extend(explanations)
            
            # Metacognitive reflection
            if len(self.thought_stream) > 10:
                reflection = self._reflect_on_thinking()
                if reflection:
                    new_thoughts.append(reflection)
            
            thoughts_generated.extend(new_thoughts)
            current_thoughts = new_thoughts
            
            if not new_thoughts:
                break  # No new thoughts generated
        
        # Update thought stream and working memory
        self.thought_stream.extend(thoughts_generated)
        
        # Keep only most relevant thoughts in working memory
        self._update_working_memory(thoughts_generated)
        
        return thoughts_generated
    
    def plan(self, goal: Goal) -> List[Thought]:
        """
        Create a plan to achieve the given goal.
        
        Uses hierarchical task decomposition and means-ends analysis
        to generate action sequences.
        """
        plan_thoughts = []
        
        # Decompose goal into subgoals
        subgoals = self._decompose_goal(goal)
        
        for subgoal in subgoals:
            # Find actions to achieve subgoal
            actions = self._find_actions_for_goal(subgoal)
            
            plan_thought = Thought(
                type=ThoughtType.PLANNING,
                content={'goal': subgoal.description, 'actions': actions},
                confidence=self._estimate_plan_success(actions),
                metadata={'goal_id': subgoal.id}
            )
            plan_thoughts.append(plan_thought)
        
        # Order plan thoughts by dependencies
        ordered_plan = self._order_by_dependencies(plan_thoughts)
        
        return ordered_plan
    
    def learn(self, experience: Dict[str, Any], outcome: str) -> None:
        """
        Learn from experience to improve future performance.
        
        Updates knowledge graph, adjusts confidence in hypotheses,
        and stores episodic memories.
        """
        # Create episodic memory
        memory = Memory(
            type=MemoryType.EPISODIC,
            content={'experience': experience, 'outcome': outcome},
            context={'goals': [g.id for g in self.current_goals]}
        )
        self.long_term_memory[MemoryType.EPISODIC].append(memory)
        
        # Extract concepts and relations for knowledge graph
        concepts = self._extract_concepts(experience)
        for concept in concepts:
            self.knowledge_graph.add_concept(concept['name'], concept.get('attributes'))
        
        # Update confidence in relevant hypotheses
        relevant_thoughts = self._find_relevant_thoughts(experience)
        for thought in relevant_thoughts:
            if thought.type == ThoughtType.HYPOTHESIS:
                # Adjust confidence based on outcome
                if outcome == 'success':
                    thought.confidence = min(1.0, thought.confidence * 1.1)
                else:
                    thought.confidence = max(0.1, thought.confidence * 0.9)
        
        # Consolidate memories if needed
        if len(self.long_term_memory[MemoryType.EPISODIC]) > 100:
            self._consolidate_memories()
    
    def _compute_salience(self, stimulus: Dict[str, Any]) -> float:
        """Compute salience of stimulus based on goals and novelty."""
        salience = 0.5  # Base salience
        
        # Increase salience if relevant to current goals
        for goal in self.current_goals:
            if self._is_relevant_to_goal(stimulus, goal):
                salience += 0.2
        
        # Increase salience for novel stimuli
        if self._is_novel(stimulus):
            salience += 0.1
        
        return min(1.0, salience)
    
    def _store_memory(self, memory_type: MemoryType, content: Any) -> None:
        """Store content in specified memory system."""
        memory = Memory(type=memory_type, content=content)
        self.long_term_memory[memory_type].append(memory)
        
        # Limit memory size
        max_size = {
            MemoryType.SENSORY: 10,
            MemoryType.WORKING: 7,
            MemoryType.EPISODIC: 1000,
            MemoryType.SEMANTIC: 10000,
            MemoryType.PROCEDURAL: 100,
            MemoryType.PROSPECTIVE: 50
        }
        
        if len(self.long_term_memory[memory_type]) > max_size[memory_type]:
            # Remove weakest memories
            memories = self.long_term_memory[memory_type]
            memories.sort(key=lambda m: m.decay(datetime.now()))
            self.long_term_memory[memory_type] = memories[-(max_size[memory_type]):]
    
    def _get_applicable_rules(self) -> List[Callable]:
        """Get reasoning rules applicable to current context."""
        # In a real implementation, these would be learned or programmed rules
        def transitivity_rule(premises):
            # If A->B and B->C then A->C
            if len(premises) >= 2:
                for i, p1 in enumerate(premises):
                    for p2 in premises[i+1:]:
                        if isinstance(p1, dict) and isinstance(p2, dict):
                            if p1.get('target') == p2.get('source'):
                                return {
                                    'source': p1.get('source'),
                                    'target': p2.get('target'),
                                    'relation': 'implies'
                                }
            return None
        
        def conjunction_rule(premises):
            # If A and B then A∧B
            if len(premises) >= 2:
                return {'conjunction': premises}
            return None
        
        return [transitivity_rule, conjunction_rule]
    
    def _retrieve_relevant_knowledge(self, thought: Thought) -> List[Thought]:
        """Retrieve knowledge relevant to given thought."""
        relevant = []
        
        # Search through semantic memory
        for memory in self.long_term_memory[MemoryType.SEMANTIC]:
            if self._compute_relevance(thought.content, memory.content) > 0.5:
                retrieved_thought = Thought(
                    type=ThoughtType.HYPOTHESIS,
                    content=memory.content,
                    confidence=memory.strength,
                    metadata={'retrieved_from': 'semantic_memory'}
                )
                relevant.append(retrieved_thought)
                memory.reinforce()  # Strengthen retrieved memory
        
        return relevant[:5]  # Return top 5 most relevant
    
    def _reflect_on_thinking(self) -> Optional[Thought]:
        """
        Engage in metacognitive reflection on recent thinking.
        
        Analyzes patterns in thought stream to identify biases,
        errors, or opportunities for improvement.
        """
        recent_thoughts = self.thought_stream[-20:]
        
        # Analyze thought patterns
        thought_types = [t.type for t in recent_thoughts]
        type_counts = {t: thought_types.count(t) for t in set(thought_types)}
        
        # Check for cognitive biases
        if type_counts.get(ThoughtType.HYPOTHESIS, 0) > 10:
            return Thought(
                type=ThoughtType.METACOGNITION,
                content="Generating many hypotheses - risk of overthinking",
                confidence=0.7,
                metadata={'pattern': 'excessive_hypothesizing'}
            )
        
        # Check for stuck patterns
        unique_contents = len(set(str(t.content) for t in recent_thoughts))
        if unique_contents < 5:
            return Thought(
                type=ThoughtType.METACOGNITION,
                content="Thinking in circles - need new perspective",
                confidence=0.8,
                metadata={'pattern': 'repetitive_thinking'}
            )
        
        return None
    
    def _update_working_memory(self, new_thoughts: List[Thought]) -> None:
        """Update working memory with most relevant thoughts."""
        # Score thoughts by relevance to goals and recency
        scored_thoughts = []
        
        for thought in new_thoughts + list(self.working_memory):
            score = thought.confidence  # Base score is confidence
            
            # Boost score for goal-relevant thoughts
            for goal in self.current_goals:
                if self._is_relevant_to_goal(thought.content, goal):
                    score += 0.2
            
            # Recency bonus
            age = (datetime.now() - thought.timestamp).total_seconds()
            score += max(0, 1 - age / 3600)  # Decay over an hour
            
            scored_thoughts.append((score, thought))
        
        # Keep top thoughts within working memory capacity
        scored_thoughts.sort(key=lambda x: x[0], reverse=True)
        self.working_memory.clear()
        
        for score, thought in scored_thoughts[:7]:  # Working memory capacity
            self.working_memory.append(thought)
    
    def _decompose_goal(self, goal: Goal) -> List[Goal]:
        """Decompose goal into subgoals."""
        subgoals = []
        
        # Simple heuristic decomposition
        # In practice, this would use more sophisticated planning
        components = goal.description.lower().split(' and ')
        
        for i, component in enumerate(components):
            subgoal = Goal(
                description=component.strip(),
                priority=goal.priority,
                parent_goal=goal.id,
                deadline=goal.deadline
            )
            subgoals.append(subgoal)
            goal.subgoals.append(subgoal.id)
        
        if not subgoals:
            # If no decomposition, return original goal
            subgoals = [goal]
        
        return subgoals
    
    def _find_actions_for_goal(self, goal: Goal) -> List[str]:
        """Find sequence of actions to achieve goal."""
        actions = []
        
        # Search procedural memory for relevant procedures
        for memory in self.long_term_memory[MemoryType.PROCEDURAL]:
            if isinstance(memory.content, dict) and 'goal' in memory.content:
                if self._is_similar_goal(memory.content['goal'], goal.description):
                    actions.extend(memory.content.get('actions', []))
                    memory.reinforce()
        
        # If no learned procedure, generate basic actions
        if not actions:
            actions = [f"Action: Work towards {goal.description}"]
        
        return actions
    
    def _estimate_plan_success(self, actions: List[str]) -> float:
        """Estimate probability of plan success."""
        if not actions:
            return 0.1
        
        # Simple heuristic: fewer actions = higher success probability
        base_probability = 0.9
        decay_per_action = 0.1
        
        return max(0.1, base_probability - (len(actions) * decay_per_action))
    
    def _order_by_dependencies(self, plan_thoughts: List[Thought]) -> List[Thought]:
        """Order plan thoughts considering dependencies."""
        # For simplicity, return as-is
        # In practice, would use topological sorting
        return plan_thoughts
    
    def _extract_concepts(self, experience: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract concepts from experience for knowledge graph."""
        concepts = []
        
        for key, value in experience.items():
            if isinstance(value, str) and len(value) > 2:
                concepts.append({
                    'name': value,
                    'attributes': {'source': 'experience', 'type': key}
                })
        
        return concepts
    
    def _find_relevant_thoughts(self, experience: Dict[str, Any]) -> List[Thought]:
        """Find thoughts relevant to given experience."""
        relevant = []
        
        for thought in self.thought_stream[-50:]:  # Check recent thoughts
            if self._compute_relevance(thought.content, experience) > 0.5:
                relevant.append(thought)
        
        return relevant
    
    def _consolidate_memories(self) -> None:
        """
        Consolidate memories by extracting patterns and creating semantic memories.
        
        Similar to sleep consolidation in biological systems.
        """
        episodic_memories = self.long_term_memory[MemoryType.EPISODIC]
        
        # Find common patterns
        patterns = defaultdict(list)
        for memory in episodic_memories[-100:]:  # Recent memories
            if isinstance(memory.content, dict):
                for key, value in memory.content.items():
                    patterns[key].append(value)
        
        # Create semantic memories from patterns
        for key, values in patterns.items():
            if len(values) >= 5:  # Minimum support
                # Find most common value
                most_common = max(set(values), key=values.count)
                semantic_memory = Memory(
                    type=MemoryType.SEMANTIC,
                    content={key: most_common, 'frequency': values.count(most_common)},
                    strength=0.8
                )
                self.long_term_memory[MemoryType.SEMANTIC].append(semantic_memory)
    
    def _is_relevant_to_goal(self, content: Any, goal: Goal) -> bool:
        """Check if content is relevant to goal."""
        if isinstance(content, dict):
            content_str = str(content)
        else:
            content_str = str(content)
        
        goal_keywords = goal.description.lower().split()
        content_keywords = content_str.lower().split()
        
        overlap = set(goal_keywords) & set(content_keywords)
        return len(overlap) > 0
    
    def _is_novel(self, stimulus: Dict[str, Any]) -> bool:
        """Check if stimulus is novel compared to recent experience."""
        for memory in self.long_term_memory[MemoryType.SENSORY][-10:]:
            if memory.content == stimulus:
                return False
        return True
    
    def _compute_relevance(self, content1: Any, content2: Any) -> float:
        """Compute relevance between two pieces of content."""
        # Simplified relevance computation
        str1 = str(content1).lower()
        str2 = str(content2).lower()
        
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)  # Jaccard similarity
    
    def _is_similar_goal(self, goal1: str, goal2: str) -> bool:
        """Check if two goals are similar."""
        return self._compute_relevance(goal1, goal2) > 0.6


class AutonomousAgent:
    """
    The main autonomous agent that combines all cognitive components
    with action execution capabilities.
    """
    
    def __init__(self, name: str = "Agent"):
        self.name = name
        self.cognitive_architecture = CognitiveArchitecture()
        self.tools = {}
        self.state = "idle"
        self.current_task = None
        self.execution_history = []
        
        # Initialize state machine for behavior control
        self.machine = Machine(
            model=self,
            states=['idle', 'perceiving', 'thinking', 'planning', 'acting', 'learning'],
            initial='idle',
            auto_transitions=False
        )
        
        # Define state transitions
        self.machine.add_transition('perceive', 'idle', 'perceiving')
        self.machine.add_transition('think', 'perceiving', 'thinking')
        self.machine.add_transition('plan', 'thinking', 'planning')
        self.machine.add_transition('act', 'planning', 'acting')
        self.machine.add_transition('learn', 'acting', 'learning')
        self.machine.add_transition('reset', '*', 'idle')
        
        # Register default tools
        self.register_tool(WebSearchTool())
        self.register_tool(CalculatorTool())
        
        logger.info(f"Agent {name} initialized")
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the complete cognitive cycle.
        
        Implements the perception-cognition-action loop with
        learning from outcomes.
        """
        result = {'status': 'processing', 'thoughts': [], 'actions': []}
        
        try:
            # Perception phase
            self.perceive()
            observation = self.cognitive_architecture.perceive(input_data)
            result['thoughts'].append(observation.to_dict())
            
            # Thinking phase
            self.think()
            thoughts = self.cognitive_architecture.think(depth=3)
            result['thoughts'].extend([t.to_dict() for t in thoughts])
            
            # Planning phase if goal-directed
            if 'goal' in input_data:
                self.plan()
                goal = Goal(description=input_data['goal'])
                self.cognitive_architecture.current_goals.append(goal)
                
                plan_thoughts = self.cognitive_architecture.plan(goal)
                result['thoughts'].extend([t.to_dict() for t in plan_thoughts])
                
                # Execute plan
                self.act()
                for plan_thought in plan_thoughts:
                    if 'actions' in plan_thought.content:
                        for action in plan_thought.content['actions']:
                            action_result = await self._execute_action(action)
                            result['actions'].append(action_result)
            
            # Learning phase
            self.learn()
            outcome = 'success' if result['actions'] else 'observation_only'
            self.cognitive_architecture.learn(input_data, outcome)
            
            result['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        finally:
            self.reset()
        
        return result
    
    async def _execute_action(self, action: str) -> Dict[str, Any]:
        """
        Execute an action using available tools.
        
        Parses action description to determine which tool to use
        and with what parameters.
        """
        action_result = {'action': action, 'status': 'unknown', 'result': None}
        
        # Parse action to determine tool and parameters
        action_lower = action.lower()
        
        if 'search' in action_lower:
            # Extract search query
            query = action.replace('Action:', '').replace('search', '').strip()
            if query and 'web_search' in self.tools:
                tool = self.tools['web_search']
                try:
                    result = await tool.execute(query)
                    action_result['status'] = 'success'
                    action_result['result'] = result
                except Exception as e:
                    action_result['status'] = 'error'
                    action_result['error'] = str(e)
        
        elif 'calculate' in action_lower or 'compute' in action_lower:
            # Extract expression
            expression = action.replace('Action:', '').replace('calculate', '').replace('compute', '').strip()
            if expression and 'calculator' in self.tools:
                tool = self.tools['calculator']
                try:
                    result = await tool.execute(expression)
                    action_result['status'] = 'success'
                    action_result['result'] = result
                except Exception as e:
                    action_result['status'] = 'error'
                    action_result['error'] = str(e)
        
        else:
            # Unknown action type
            action_result['status'] = 'unrecognized'
        
        # Store in execution history
        self.execution_history.append(action_result)
        
        return action_result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status including cognitive state."""
        working_memory_contents = [
            t.to_dict() for t in self.cognitive_architecture.working_memory
        ]
        
        goal_status = [
            {'description': g.description, 'progress': g.progress, 'status': g.status}
            for g in self.cognitive_architecture.current_goals
        ]
        
        return {
            'name': self.name,
            'state': self.state,
            'working_memory': working_memory_contents,
            'goals': goal_status,
            'available_tools': list(self.tools.keys()),
            'thought_count': len(self.cognitive_architecture.thought_stream),
            'execution_history_size': len(self.execution_history)
        }
    
    def visualize_knowledge_graph(self) -> None:
        """Create visual representation of knowledge graph."""
        kg = self.cognitive_architecture.knowledge_graph
        
        if len(kg.graph) == 0:
            console.print("[yellow]Knowledge graph is empty[/yellow]")
            return
        
        tree = Tree("Knowledge Graph")
        
        # Get important concepts
        importance = kg.compute_importance()
        sorted_concepts = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for concept, score in sorted_concepts[:10]:
            branch = tree.add(f"{concept} (importance: {score:.3f})")
            
            # Add related concepts
            related = kg.get_related_concepts(concept, max_distance=1)
            for related_concept in list(related)[:3]:
                branch.add(f"→ {related_concept}")
        
        console.print(tree)
    
    def introspect(self) -> None:
        """
        Agent introspection - analyze and report on internal state.
        
        Provides insights into the agent's thinking patterns, memory
        usage, and goal progress.
        """
        panel_content = []
        
        # Analyze thought patterns
        thought_types = defaultdict(int)
        confidence_sum = defaultdict(float)
        
        for thought in self.cognitive_architecture.thought_stream:
            thought_types[thought.type.name] += 1
            confidence_sum[thought.type.name] += thought.confidence
        
        panel_content.append("[bold]Thought Analysis:[/bold]")
        for thought_type, count in thought_types.items():
            avg_confidence = confidence_sum[thought_type] / count if count > 0 else 0
            panel_content.append(f"  {thought_type}: {count} thoughts (avg confidence: {avg_confidence:.2f})")
        
        # Memory utilization
        panel_content.append("\n[bold]Memory Systems:[/bold]")
        for memory_type, memories in self.cognitive_architecture.long_term_memory.items():
            panel_content.append(f"  {memory_type.name}: {len(memories)} memories")
        
        # Goal progress
        if self.cognitive_architecture.current_goals:
            panel_content.append("\n[bold]Goal Progress:[/bold]")
            for goal in self.cognitive_architecture.current_goals[:3]:
                panel_content.append(f"  {goal.description}: {goal.progress*100:.0f}% complete")
        
        # Metacognitive insights
        recent_metacognition = [
            t for t in self.cognitive_architecture.thought_stream
            if t.type == ThoughtType.METACOGNITION
        ]
        
        if recent_metacognition:
            panel_content.append("\n[bold]Self-Reflection:[/bold]")
            for thought in recent_metacognition[-3:]:
                panel_content.append(f"  • {thought.content}")
        
        panel = Panel("\n".join(panel_content), title=f"[cyan]{self.name} Introspection[/cyan]")
        console.print(panel)


async def demo_agent():
    """Demonstrate the autonomous agent capabilities."""
    console.print("[bold cyan]Autonomous AI Agent Demo[/bold cyan]\n")
    
    # Create agent
    agent = AutonomousAgent(name="Prometheus")
    
    # Process various inputs
    test_inputs = [
        {
            'observation': 'The weather is sunny and warm',
            'goal': 'Plan outdoor activity'
        },
        {
            'observation': 'Stock market showed 5% gain today',
            'goal': 'Analyze investment opportunities'
        },
        {
            'observation': 'User seems frustrated with slow response',
            'goal': 'Improve user experience'
        }
    ]
    
    for input_data in test_inputs:
        console.print(f"\n[green]Input:[/green] {input_data}")
        
        result = await agent.process_input(input_data)
        
        # Display results
        if result['thoughts']:
            console.print("\n[yellow]Generated Thoughts:[/yellow]")
            for thought in result['thoughts'][-3:]:  # Show last 3 thoughts
                console.print(f"  • [{thought['type']}] {thought['content']} (confidence: {thought['confidence']:.2f})")
        
        if result['actions']:
            console.print("\n[blue]Executed Actions:[/blue]")
            for action in result['actions']:
                console.print(f"  • {action['action']} -> {action['status']}")
    
    # Show agent introspection
    console.print("\n")
    agent.introspect()
    
    # Visualize knowledge graph
    console.print("\n")
    agent.visualize_knowledge_graph()
    
    # Show final status
    console.print("\n[bold]Final Agent Status:[/bold]")
    status = agent.get_status()
    
    table = Table(title="Agent Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("State", status['state'])
    table.add_row("Thoughts Generated", str(status['thought_count']))
    table.add_row("Working Memory Items", str(len(status['working_memory'])))
    table.add_row("Active Goals", str(len(status['goals'])))
    table.add_row("Actions Executed", str(status['execution_history_size']))
    
    console.print(table)


if __name__ == "__main__":
    # Run the demonstration
    console.print(Panel.fit(
        "[bold]Advanced Autonomous AI Agent System[/bold]\n\n"
        "This agent features:\n"
        "• Cognitive architecture with multiple memory systems\n"
        "• Various reasoning strategies (deductive, inductive, abductive, analogical)\n"
        "• Goal-directed planning and execution\n"
        "• Metacognitive self-reflection\n"
        "• Knowledge graph representation\n"
        "• Tool use and action execution\n"
        "• Learning from experience\n",
        title="Prometheus Agent Framework",
        border_style="cyan"
    ))
    
    asyncio.run(demo_agent())
