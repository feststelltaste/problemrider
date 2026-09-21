---
title: Bewährte Technologien
description: Nutzung erprobter und ausgereifter Technologien.
category:
- Architecture
- Process
quality_tactics_url: https://qualitytactics.de/en/reliability/boring-technologies/
problems:
- cv-driven-development
- cargo-culting
- second-system-effect
- gold-plating
- rapid-prototyping-becoming-production
- assumption-based-development
- suboptimal-solutions
- insufficient-design-skills
- implementation-rework
- premature-technology-introduction
- technology-isolation
layout: solution
lang: de
en_slug: boring-technologies
related_solutions:
- slug: technology-radar
  similarity: 0.8
- slug: architecture-decision-records
  similarity: 0.75
- slug: technical-debt-backlog
  similarity: 0.75
- slug: strangler-fig-pattern
  similarity: 0.75
- slug: architecture-roadmap
  similarity: 0.7
- slug: ci-cd-pipeline
  similarity: 0.7
---

## Description

Die Wahl bewährter Technologie bedeutet, bewusst Werkzeuge zu bevorzugen, deren Fehlermodi bereits gut verstanden sind und in denen das Team bereits genuine Expertise hat, gegenüber neuartigen Alternativen, die wegen ihrer Attraktivität statt ihrer Passung gewählt werden — es bedeutet nicht, veraltete Technologie zu wählen. Legacy-Modernisierungsbemühungen sind besonders anfällig für die gegenteilige Versuchung: Eine Neuschreibung ist genau der Moment, in dem ein unvertrautes Framework, eine Datenbank oder ein Architekturmuster am attraktivsten aussieht, und genau der Moment, in dem sich das Team die von ihm eingeführten Unbekannten am wenigsten leisten kann. Eine explizite „Bewährte-Alternative"-Analyse zu verlangen, bevor irgendetwas Neues übernommen wird, und Entscheidungen gegen die tatsächliche Expertise des Teams statt gegen Branchentrends zu bewerten, hält Neuartigkeit davon ab, um ihrer selbst willen übernommen zu werden, während weiterhin Raum für genuin gerechtfertigte Veränderung bleibt.

## How to Apply ◆

> Im Kontext von Legacy-Systemen bedeutet „bewährte Technologie" nicht veraltete Technologie — es bedeutet die Wahl von Werkzeugen und Ansätzen, bei denen die Fehlermodi gut verstanden sind, das Team genuine Expertise hat und die operativen Kosten vorhersagbar sind. Dies wirkt direkt der Tendenz entgegen, glänzende Lösungen zu übernehmen, die neue Probleme schaffen, während sie versuchen, alte zu lösen.

- Etablieren Sie einen Technologie-Radar für Ihre Organisation, der Technologien explizit in „übernehmen", „testen", „bewerten" und „zurückhalten" kategorisiert. Machen Sie „übernehmen" zum Standard für Produktivsysteme und verlangen Sie eine schriftliche Rechtfertigung — gebunden an ein spezifisches Geschäftsbedürfnis, nicht an Entwicklerinteresse —, bevor irgendetwas aus den anderen Kategorien eingeführt wird.
- Verlangen Sie eine „Bewährte-Alternative-Analyse" für jeden Technologievorschlag: Bevor ein neues Framework, eine Datenbank oder ein Architekturmuster übernommen wird, muss der Vorschlagende dokumentieren, was die bewährte, gut verstandene Alternative wäre und warum sie unzureichend ist. Dies erzwingt ehrliche Bewertung, ob Neuartigkeit um ihrer selbst willen gewählt wird.
- Wenden Sie das Konzept der „Innovations-Tokens" an: Jedes Team oder Projekt hat ein begrenztes Budget an Komplexität, das für neuartige Technologien oder Ansätze ausgegeben werden kann. Sobald das Budget ausgegeben ist, muss jede verbleibende technische Entscheidung bewährte, gut verstandene Werkzeuge nutzen. Dies verhindert die kumulative Komplexitätsexplosion, die aus der gleichzeitigen Übernahme mehrerer unvertrauter Technologien resultiert.
- Bewerten Sie Technologiewahlen gegen die Expertise des Teams, nicht gegen Branchentrends. Eine Technologie, die bei einem Unternehmen mit 500 Ingenieuren und einem dedizierten Plattformteam brillant funktioniert, könnte für ein Team von acht Personen, die ein Legacy-System pflegen, eine Katastrophe sein. Fragen Sie „kann unser Team das um 3 Uhr morgens debuggen?" statt „ist das, was moderne Unternehmen nutzen?"
- Verhindern Sie, dass Prototypencode Produktion erreicht, indem Sie ein klares Prototyp-zu-Produktion-Gate etablieren. Mit experimentellen Technologien gebaute Prototypen erfüllen ihren Zweck — Machbarkeit demonstrieren —, aber die Produktionsimplementierung sollte den zuverlässigsten Stack des Teams nutzen, es sei denn, es gibt einen zwingenden technischen Grund dagegen.
- Bekämpfen Sie den Second-System-Effekt, indem Sie einen minimal lebensfähigen Ersatzumfang definieren, bevor das Design beginnt, und ihn durch regelmäßige Umfangsüberprüfungen durchsetzen. Beim Ersatz eines Legacy-Systems muss der Tendenz, jedes Feature hinzuzufügen, das dem alten System fehlte, aktiv widerstanden werden, indem validierte Nutzernachfrage für jede vorgeschlagene Fähigkeit verlangt wird.
- Treffen Sie Technologieentscheidungen als Team, statt einzelnen Entwicklern zu erlauben, einseitig neue Werkzeuge einzuführen. Gruppenentscheidungsfindung filtert natürlich CV-getriebene Entscheidungen heraus, weil der Vorschlagende Kollegen überzeugen muss, die die Technologie auch pflegen müssen.
- Dokumentieren Sie alle Technologieentscheidungen in Architecture Decision Records (ADRs), die Kontext, erwogene Optionen, Entscheidungsbegründung und erwartete Konsequenzen beinhalten. Dies schafft Verantwortlichkeit und macht sichtbar, wenn Entscheidungen von Lebenslaufaufbau statt von Projektbedürfnissen getrieben wurden.

## Tradeoffs ⇄

> Bewährte Technologien verringern Überraschung und operatives Risiko, erfordern aber die Disziplin, der Anziehungskraft von Neuartigkeit zu widerstehen, und die Reife zu akzeptieren, dass die Lösung von Problemen mit gut verstandenen Werkzeugen wertvoller ist als ihre Lösung mit beeindruckenden.

**Vorteile:**

- Eliminiert die Wissenslücke, die entsteht, wenn CV-getriebene Technologiewahlen das Team unfähig zurücklassen, Systeme zu pflegen, nachdem der ursprüngliche Entwickler weitergezogen ist, weil das gesamte Team bereits im gewählten Stack versiert ist.
- Verringert Implementierungs-Rework durch Vermeidung von Technologien, deren Beschränkungen erst nach erheblichem Entwicklungsaufwand sichtbar werden, da bewährte Technologien gut dokumentierte Beschränkungen und Workarounds haben.
- Verhindert das Cargo-Cult-Antimuster, indem Teams verlangt wird zu verstehen, warum eine Technologie für ihren Kontext angemessen ist, statt sie zu übernehmen, weil erfolgreiche Unternehmen sie nutzen.
- Senkt operative Kosten, weil das Team Produktionsprobleme mit seiner bestehenden Expertise beheben kann, statt Debugging-Techniken für unvertraute Werkzeuge unter Vorfalldruck zu lernen.
- Wirkt Gold Plating und dem Second-System-Effekt entgegen, indem der Lösungsraum auf bewährte Ansätze beschränkt wird, was es schwieriger macht, unnötige Komplexität oder spekulative Features zu rechtfertigen.

**Kosten und Risiken:**

- Ins Extreme getrieben, kann „bewährte Technologien" zu einer Ausrede für technologische Stagnation werden, bei der Teams nie genuin vorteilhafte Innovationen übernehmen, weil jede Änderung als riskant wahrgenommen wird.
- Teams, die nur vertraute Werkzeuge nutzen, könnten erhebliche Produktivitäts- oder Zuverlässigkeitsverbesserungen verpassen, die neuere Technologien bieten, die genug gereift sind, um bei anderen Organisationen als „bewährt" zu gelten.
- Talentierte Entwickler, die Lerngelegenheiten schätzen, könnten frustriert werden, wenn die Richtlinie als Innovationshemmung wahrgenommen wird, was möglicherweise die Fluktuation erhöht — die Richtlinie muss kontrolliertes Experimentieren in nicht-kritischen Kontexten erlauben.
- Die Definition von „bewährt" ist subjektiv und kontextabhängig: Was für ein Team bewährt und gut verstanden ist, könnte für ein anderes neuartig und riskant sein, sodass die Richtlinie auf die tatsächliche Expertise jedes Teams kalibriert werden muss.
- Legacy-Systeme, die bereits auf inzwischen veralteten Technologien gebaut sind, könnten Migrationen zu moderneren (aber immer noch bewährten) Alternativen brauchen, und das Prinzip bewährter Technologie sollte nicht genutzt werden, um das Verbleiben auf nicht unterstützten Plattformen zu rechtfertigen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie das Prinzip bewährter Technologie angewendet wurde, um übliche Antimuster in der Legacy-Systemmodernisierung zu verhindern.

Ein Fintech-Startup, das ein Legacy-Zahlungsabwicklungssystem ersetzte, erwog den Bau des Ersatzes mit einer Event-Sourcing-Architektur, unterstützt durch Apache Kafka, einer GraphQL-API-Schicht und einer verteilten NoSQL-Datenbank — Technologien, die sich mehrere Entwickler für ihren Lebenslauf wünschten. Ein Senior-Architekt wandte die Analyse bewährter Technologie an: Das Team hatte tiefe Expertise in PostgreSQL, REST-APIs und einer Standard-Nachrichtenwarteschlange. Die bewährte Alternative konnte das Transaktionsvolumen des Unternehmens mit Spielraum handhaben. Das Team baute den Ersatz mit bewährten Technologien und lieferte ihn in vier Monaten statt der geschätzten zwölf, die der „moderne" Stack erfordert hätte. Zwei Jahre später handhabt das System das Zehnfache des ursprünglichen Volumens ohne architektonische Änderungen, und jedes Teammitglied kann Produktionsprobleme unabhängig debuggen.

Eine Regierungsbehörde, die ihr Bürgerportal neu baute, litt unter dem Second-System-Effekt: Das Design beinhaltete KI-gestützte Formularassistenz, blockchain-basierte Dokumentenverifikation, eine Microservices-Architektur mit 23 geplanten Services und Echtzeitanalytik — alles Features, die dazu gedacht waren, Frustrationen mit dem alten System zu adressieren. Nach achtzehn Monaten und erheblicher Budgetüberschreitung hatte das Team nur drei der geplanten Services geliefert. Ein Projekt-Reset wandte das Prinzip bewährter Technologie an: Das Team identifizierte die fünf meistgenutzten Bürgerinteraktionen, baute sie als Standard-Webanwendung mit relationaler Datenbank und deployte innerhalb von vier Monaten. Die verbleibende Funktionalität wurde basierend auf tatsächlichen Nutzungsdaten statt spekulativer Anforderungen priorisiert, und die meisten der ursprünglich geplanten „innovativen" Features wurden nie von tatsächlichen Nutzern angefragt.

Ein Logistikunternehmen entdeckte, dass drei verschiedene Teams unabhängig voneinander drei verschiedene Nachrichtenwarteschlangen-Technologien eingeführt hatten — RabbitMQ, Apache Kafka und Amazon SQS —, weil der leitende Entwickler jedes Teams Erfahrung mit einem anderen Werkzeug wollte. Kein einzelnes Teammitglied verstand alle drei, und teamübergreifendes Debugging von Nachrichtenflussproblemen erforderte die Zusammenstellung von Experten aus mehreren Teams. Durch Anwendung des Prinzips bewährter Technologie ordnete die Engineering-Führung die Konsolidierung auf RabbitMQ an, das die größte Anzahl von Entwicklern bereits verstand und das die technischen Anforderungen aller drei Anwendungsfälle erfüllte. Die Konsolidierung dauerte sechs Wochen, eliminierte aber eine gesamte Kategorie von Produktionsvorfällen im Zusammenhang mit falsch konfigurierten Warteschlangen-Konsumenten, und Bereitschaftsingenieure konnten nun jeden Nachrichtenfluss beheben, ohne Spezialistenwissen zu benötigen.
