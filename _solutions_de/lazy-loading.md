---
title: Lazy Loading
description: Verzögertes Laden von Daten und Ressourcen bis zum Moment der
  tatsächlichen Nutzung.
category:
- Performance
problems:
- slow-application-performance
- high-client-side-resource-consumption
- memory-leaks
- slow-response-times-for-lists
- excessive-object-allocation
- gradual-performance-degradation
- high-resource-utilization-on-client
- inefficient-frontend-code
layout: solution
lang: de
en_slug: lazy-loading
related_solutions:
- slug: lazy-evaluation
  similarity: 0.95
- slug: predictive-loading
  similarity: 0.8
- slug: progressive-loading
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: connection-pooling
  similarity: 0.8
- slug: code-splitting
  similarity: 0.8
---

## Description

Lazy Loading verzögert das Abrufen oder Initialisieren einer Ressource — eine UI-Komponente, eine Datenbankassoziation, ein Bild oder eine Datenseite — bis zu dem Punkt, an dem sie von der aktuellen Interaktion des Nutzers echt gebraucht wird, statt zur Konstruktionszeit alles abzurufen, was ein Bildschirm oder Objekt jemals brauchen könnte. Es wird implementiert durch Mechanismen wie Bundle-Splitting und dynamische Imports im Frontend, lazy initialisierte Assoziationen in einem ORM oder virtuelles Scrollen und Paginierung für große Listen, die alle dieselbe zugrundeliegende Idee teilen, eine verzögerte Referenz gegen sofortige Materialisierung einzutauschen. Legacy-Anwendungen entwickelten ihre Gewohnheiten eifrigen Ladens häufig organisch: Ein Bildschirm, der einst eine Handvoll Datensätze zeigte, rendert nun Tausende, oder eine Startroutine, die einst wenige Module initialisierte, bootet nun Dutzende Subsysteme, von denen niemand mehr weiß, dass sie ungenutzt sind, und weil nichts eine Neubewertung dieser Ladestrategie erzwang, kroch der Ressourcenverbrauch Jahr für Jahr nach oben. Lazy Loading in ein solches System einzuführen zielt direkt auf die langsamen Startzeiten, aufgeblähten Speicher-Footprints und trägen Listen-Renderings ab, die sich auf diese Weise ansammeln, ohne dass der umgebende Legacy-Code neu geschrieben werden muss — die Ladegrenze kann meist am Zugriffspunkt statt in der gesamten Codebasis eingefügt werden. Weil die verschobenen Kosten dennoch irgendwann bezahlt werden müssen, oft in einem unvorhersehbaren, für den Endnutzer sichtbaren Moment, muss Lazy Loading in Legacy-Kontexten mit klaren Ladeindikatoren und Schutzmaßnahmen gegen Muster wie N+1-Abfragen gepaart werden, bei denen eine innerhalb einer Schleife zugegriffene Lazy-Assoziation still die Zahl verzögerter Abrufe vervielfacht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Profilieren Sie die Anwendung, um eifrig geladene Ressourcen zu identifizieren, die selten oder nie in typischen Nutzerabläufen genutzt werden
- Ersetzen Sie eifrige Initialisierung schwergewichtiger Objekte durch Lazy Proxies oder Factory-Methoden, die die Erstellung verschieben
- Implementieren Sie Lazy Loading für UI-Komponenten, indem Bundles aufgeteilt und bei Bedarf geladen werden
- Wandeln Sie Datenbankabfragen, die ganze Objektgraphen abrufen, in Abfragen um, die Beziehungen nur bei Zugriff laden
- Nutzen Sie framework-spezifische Lazy-Loading-Features (z. B. ORM-Lazy-Assoziationen, React.lazy, dynamische Imports), wo verfügbar
- Fügen Sie Monitoring hinzu, um tatsächliche Ressourcennutzungsmuster zu verfolgen und zu validieren, dass verzögerte Ressourcen geladen werden, wenn sie echt gebraucht werden
- Stellen Sie sicher, dass die Fehlerbehandlung Fälle abdeckt, in denen lazy geladene Ressourcen zum Zeitpunkt der tatsächlichen Nutzung nicht verfügbar werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert anfängliche Ladezeit und Speicher-Footprint, was die gefühlte Performance verbessert
- Verringert Ressourcenverbrauch für Features, auf die Nutzer in einer gegebenen Sitzung nie zugreifen
- Erlaubt Legacy-Systemen, größere Datensätze zu handhaben, ohne Infrastruktur-Upgrades zu erfordern
- Verbessert die Startzeit für monolithische Anwendungen mit vielen Subsystemen

**Kosten und Risiken:**
- Führt Latenz beim ersten Zugriff ein, was Nutzer überraschen kann, wenn nicht mit Ladeindikatoren gehandhabt
- Fügt Komplexität zur Initialisierungslogik hinzu und kann schwer zu debuggende Reihenfolgeprobleme erzeugen
- Kann N+1-Abfrageprobleme in ORMs verursachen, wenn Lazy-Assoziationen in Schleifen zugegriffen werden
- Verkompliziert das Testen, weil Verhalten davon abhängt, wann Ressourcen tatsächlich geladen werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Enterprise-Resource-Planning-System lud alle Referenzdatentabellen beim Start in den Speicher, was eine 45-sekündige Bootzeit verursachte und über 2 GB RAM verbrauchte. Durch die Umstellung der Referenzdaten-Loader auf Lazy-Initialisierung verringerte das Team die Startzeit auf unter 8 Sekunden und halbierte den Basis-Speicherverbrauch. Selten genutzte Module wie Archiv-Reporting und Audit-Historie wurden nur geladen, wenn Nutzer zu diesen Bereichen navigierten, was auch den Explosionsradius von Bugs in diesen Subsystemen während des normalen Betriebs verringerte.
